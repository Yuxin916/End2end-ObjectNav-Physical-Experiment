from internvl_cleaned.constants_utils import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import io
import re
import transformers
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from internvl_cleaned.constants_utils import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD)
from internvl_cleaned.conversation import get_conv_template
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


from transformers.utils import logging

logger = logging.get_logger(__name__)


def parse_coordinate_weight_string(
    raw: Optional[str],
    expected_len: int,
    default_values: List[float],
    field_name: str,
    allowed_lens: Optional[List[int]] = None,
) -> List[float]:
    if raw is None or raw.strip() == '':
        return list(default_values)

    cleaned = raw.strip()
    if cleaned.startswith('[') and cleaned.endswith(']'):
        cleaned = cleaned[1:-1]

    parts = [item.strip() for item in cleaned.split(',') if item.strip() != '']
    try:
        values = [float(item) for item in parts]
    except ValueError as exc:
        raise ValueError(
            f'Invalid {field_name}: {raw}. Please provide comma-separated numeric values.'
        ) from exc

    valid_lens = allowed_lens if allowed_lens is not None else [expected_len]
    if len(values) not in valid_lens:
        lens_text = ','.join(str(v) for v in valid_lens)
        raise ValueError(
            f'Invalid {field_name}: expected one of lengths [{lens_text}], got {len(values)} ({raw}).'
        )
    return values


def build_coordinate_answer_weights(
    answer_text: str,
    answer_token_len: int,
    weights_with_space: List[float],
) -> np.ndarray:
    """
    Build per-token weights for answers formatted as "XXX, YYY".
    Supports either 8 weights (coordinate only) or 9 weights (coordinate + EOS/im_end).
    Falls back to all-zero weights if format/token length does not match expectations.
    """
    normalized = answer_text.strip()
    if not re.fullmatch(r'\d{3}, \d{3}', normalized):
        return np.zeros(answer_token_len, dtype=np.float32)

    position_weights = weights_with_space
    if len(position_weights) not in (8, 9):
        return np.zeros(answer_token_len, dtype=np.float32)

    # The coordinate text itself should tokenize into exactly 8 tokens: XXX, YYY.
    if answer_token_len != 8:
        return np.zeros(answer_token_len, dtype=np.float32)

    return np.array(position_weights[:8], dtype=np.float32)



def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):

    # Mean and std normalization
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError

    if is_train:  # use data argumentation in meta file
        # in training, images are slightly randomized each epoch to teach the model invariance to visual distortions
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            # pads shorter sides of the image with a constant color (mean pixel value) to make it square without distortion
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform



def preprocess_internvl2_5(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        use_coordinate_loss_weight: bool = False,
        coordinate_loss_weights_with_space: Optional[str] = '3,2,1,0,0,3,2,1',
) -> Dict:
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]
    parsed_weights_with_space = parse_coordinate_weight_string(
        coordinate_loss_weights_with_space,
        expected_len=8,
        default_values=[3.0, 2.0, 1.0, 0.0, 0.0, 3.0, 2.0, 1.0],
        field_name='coordinate_loss_weights_with_space',
        allowed_lens=[8, 9],
    )

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message
        # system_prompt = None

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}'
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

    batches, roles, assistant_texts = [], [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
        assistant_texts.append(None)
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('human')
            assistant_texts.append(None)
        elif conversation['from'] == 'gpt':
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append('gpt')
            assistant_texts.append(conversation['value'])
        elif conversation['from'] == 'pixel_label':
            continue
        elif conversation['from'] == 'traversability_mask':
            continue
        elif isinstance(conversation['from'], str) and conversation['from'].lower() == 'traj_mask':
            continue
        else:
            raise NotImplementedError

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets, final_loss_weights = [], [], []
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    for role, input_id, assistant_text in zip(roles, input_ids, assistant_texts):
        final_input_ids.append(input_id)
        if role == 'system' or role == 'human':
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
            final_loss_weights.append(np.zeros(input_id.shape, dtype=np.float32))
        elif role == 'gpt':
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_targets.append(target)

            loss_weight = np.zeros(input_id.shape, dtype=np.float32)
            answer_ids = tokenizer(
                assistant_text,
                return_tensors='np',
                padding=False,
                truncation=False,
                add_special_tokens=False,
            ).input_ids[0]
            if use_coordinate_loss_weight:
                answer_weights = build_coordinate_answer_weights(
                    assistant_text,
                    answer_ids.shape[0],
                    weights_with_space=parsed_weights_with_space,
                )
            else:
                answer_weights = np.ones(answer_ids.shape[0], dtype=np.float32)

            answer_start = ignore_len
            answer_end = min(answer_start + answer_ids.shape[0], input_id.shape[0])
            if answer_end > answer_start:
                active_len = answer_end - answer_start
                loss_weight[answer_start:answer_end] = answer_weights[:active_len]

            # Optional 9th weight supervises the token right after "XXX, YYY" if it is EOS/im_end.
            if use_coordinate_loss_weight and len(parsed_weights_with_space) == 9:
                eos_pos = answer_start + answer_ids.shape[0]
                if eos_pos < input_id.shape[0]:
                    eos_token_id = tokenizer.eos_token_id
                    im_end_ids = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
                    im_end_token_id = im_end_ids[0] if len(im_end_ids) == 1 else None
                    next_id = int(input_id[eos_pos])
                    if (eos_token_id is not None and next_id == eos_token_id) or (
                        im_end_token_id is not None and next_id == im_end_token_id
                    ):
                        loss_weight[eos_pos] = float(parsed_weights_with_space[8])

            loss_weight[target == IGNORE_TOKEN_ID] = 0.0
            final_loss_weights.append(loss_weight)
        elif conversation['from'] == 'pixel_label':
            continue
        elif conversation['from'] == 'traversability_mask':
            continue
        elif isinstance(conversation['from'], str) and conversation['from'].lower() == 'traj_mask':
            continue
        else:
            raise NotImplementedError
    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]
    loss_weight = torch.tensor(np.concatenate(final_loss_weights), dtype=torch.float32)[:tokenizer.model_max_length]

    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)
        loss_weight = F.pad(loss_weight, (0, padding_length), value=0.0)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)
    loss_weight = loss_weight.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        loss_weight=loss_weight,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
