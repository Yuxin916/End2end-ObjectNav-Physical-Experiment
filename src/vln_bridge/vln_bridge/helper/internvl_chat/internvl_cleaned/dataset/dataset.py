import sys
from torch.utils.data import Dataset
import torch.distributed as dist

import json
import os
import math
import random
import re
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from copy import deepcopy
import traceback
from typing import Dict, List, Literal, Optional

from internvl_cleaned.dataset.data_preprocess import (build_transform,
                                                      dynamic_preprocess,
                                                      preprocess_internvl2_5
                                                      )
from internvl_cleaned.dataset.data_packing import PackedDataset
from internvl_cleaned.constants_utils import IMG_END_TOKEN
from transformers.utils import logging

logger = logging.get_logger(__name__)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
        use_coordinate_loss_weight=False,
        coordinate_loss_weights_with_space='3,2,1,0,0,3,2,1',
    ):
        # Initialize the dataset class
        super(LazySupervisedDataset, self).__init__()

        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode

        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1  #
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        self.use_coordinate_loss_weight = use_coordinate_loss_weight
        self.coordinate_loss_weights_with_space = coordinate_loss_weights_with_space

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            raise NotImplementedError("group_by_length is not supported in lazy mode.")

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name in {'internvl2_5', 'internvl2_5_nav'}:
            preprocess_function = preprocess_internvl2_5
        else:
            raise NotImplementedError
        return preprocess_function

    def load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def _extract_traj_mask_from_conversations(self, conversations) -> Optional[List]:
        for conversation in conversations:
            if not isinstance(conversation, dict):
                continue
            speaker = conversation.get('from', '')
            if isinstance(speaker, str) and speaker.lower() == 'traj_mask':
                return conversation.get('value', None)
        return None

    def _build_history_traj_mask(self, data_item, num_tiles: List[int], num_patches: int) -> torch.LongTensor:
        # Default: no trajectory on all visual tokens.
        history_traj_mask = torch.zeros((num_patches, self.num_image_token), dtype=torch.long)
        raw_mask = self._extract_traj_mask_from_conversations(data_item.get('conversations', []))
        if raw_mask is None:
            return history_traj_mask

        if not isinstance(raw_mask, list):
            logger.warning(f'[{self.ds_name}] Traj_MASK is not a list, fallback to zeros.')
            return history_traj_mask

        num_images = len(num_tiles)
        num_history_images = max(0, num_images - 1)

        # Accept both formats:
        # 1) len(mask) == num_images: [current, history1, history2, ...]
        # 2) len(mask) == num_history_images: [history1, history2, ...]
        use_all_images_mask = len(raw_mask) == num_images
        use_history_only_mask = len(raw_mask) == num_history_images
        if not (use_all_images_mask or use_history_only_mask):
            logger.warning(
                f'[{self.ds_name}] Traj_MASK length ({len(raw_mask)}) does not match '
                f'num_images ({num_images}) or num_history_images ({num_history_images}). '
                'Will use best-effort alignment and missing items will fallback to zeros.'
            )

        patch_start = 0
        for image_idx, n_tiles in enumerate(num_tiles):
            tile_count = int(n_tiles)
            patch_end = patch_start + tile_count
            if patch_start >= num_patches:
                break
            patch_end = min(patch_end, num_patches)

            # Skip current image (index 0): this mask is only for history images.
            if image_idx == 0:
                patch_start = patch_end
                continue

            if use_all_images_mask:
                mask_idx = image_idx
            elif use_history_only_mask:
                mask_idx = image_idx - 1
            else:
                # Best effort for legacy or mixed-quality data.
                mask_idx = image_idx if image_idx < len(raw_mask) else image_idx - 1

            image_mask = raw_mask[mask_idx] if 0 <= mask_idx < len(raw_mask) else None
            if image_mask is None:
                patch_start = patch_end
                continue

            try:
                image_mask_tensor = torch.as_tensor(image_mask, dtype=torch.long).reshape(-1)
            except Exception:
                logger.warning(f'[{self.ds_name}] Failed to parse Traj_MASK for image_idx={image_idx}, fallback to zeros.')
                patch_start = patch_end
                continue

            # Preferred format: one binary value per visual token (e.g., 16x16 -> 256).
            if image_mask_tensor.numel() == self.num_image_token:
                image_mask_tensor = (image_mask_tensor > 0).to(dtype=torch.long)
                repeated = image_mask_tensor.view(1, -1).expand(patch_end - patch_start, -1).clone()
                history_traj_mask[patch_start:patch_end] = repeated
            else:
                # Fallback: treat mask as image-level binary label and broadcast to all visual tokens.
                scalar_flag = int((image_mask_tensor > 0).any().item())
                history_traj_mask[patch_start:patch_end] = scalar_flag

            patch_start = patch_end

        return history_traj_mask

    def multi_modal_get_item(self, data_item):
        # bascially, my code never goes here:
        print("Error: single image case should not be used in current experiments. in intervl_chat/dataset/dataset.py")
        sys.exit()
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # get pixel label ---------------------------------------------------------------
        # pixel_label = self._parse_pixel_label(data_item)
        # if pixel_label is None:
        #     pixel_labels = torch.full((pixel_values.size(0), 2), -1.0, dtype=torch.float32)
        # else:
        #     pixel_labels = pixel_label.repeat(pixel_values.size(0), 1)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name,
                                  use_coordinate_loss_weight=self.use_coordinate_loss_weight,
                                  coordinate_loss_weights_with_space=self.coordinate_loss_weights_with_space)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            loss_weight=ret['loss_weight'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            # pixel_labels=pixel_labels,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            image_type_ids=torch.tensor([0] * num_patches, dtype=torch.long),
            history_traj_mask=torch.zeros((num_patches, self.num_image_token), dtype=torch.long),
        )
        return ret


    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image,
                                  use_coordinate_loss_weight=self.use_coordinate_loss_weight,
                                  coordinate_loss_weights_with_space=self.coordinate_loss_weights_with_space)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # generate pixel lables -------------------------------------------------------------------------------
        # conversations = data_item['conversations']
        # pixel_labels = None
        # traversability_labels = None
        # for i in range(len(conversations)):
        #     if 'from' in conversations[i] and conversations[i]['from'] == 'pixel_label':
        #         pixel_value = conversations[i]['value']
        #         pixel_labels = torch.tensor(pixel_value, dtype=torch.float32)
            # if 'from' in conversations[i] and conversations[i]['from'] == 'traversability_mask':
            #     traversability_path = conversations[i]['value']
            #     traversability_mask = self.load_image(self.get_image_path(traversability_path)).convert('L')
            #     traversability_mask = traversability_mask.resize((256, 256), resample=Image.NEAREST)
            #     mask_array = np.array(traversability_mask)
                # traversability_labels = (mask_array < 128).astype(np.int32)
                # traversability_labels = torch.tensor(traversability_labels, dtype=torch.int32).unsqueeze(0).unsqueeze(-1)
        # if pixel_labels is None:
        #     raise ValueError(f'No pixel label found in data_item: {data_item}')
        # if pixel_labels.dim() == 1: # ensure packed dataset always sees 2D pixel labels
        #     pixel_labels = pixel_labels.unsqueeze(0)
        # if pixel_labels.size(-1) != 2:
        #     raise ValueError(f'Pixel label should have shape (*, 2), but got {pixel_labels.shape}')
        # if traversability_labels is None:
        #     traversability_labels = torch.full((1, 256, 256, 1), -100, dtype=torch.int32)
        # else:
            # if traversability_labels.dim() == 2:
            #     traversability_labels = traversability_labels.unsqueeze(0).unsqueeze(-1)
            # elif traversability_labels.dim() == 3:
            #     traversability_labels = traversability_labels.unsqueeze(0)
            # if traversability_labels.dim() == 2:
            #     traversability_labels = traversability_labels.unsqueeze(-1)
            # if traversability_labels.shape[-3:] != (256, 256, 1):
            #     raise ValueError(
            #         f'Traversability label should have shape (*, 256, 256, 1), '
            #         f'but got {traversability_labels.shape}'
            #     )
        

        # Create the final return dictionary
        num_current_patches = num_tiles[0] if len(num_tiles) > 0 else 0
        image_type_ids = torch.ones(num_patches, dtype=torch.long)
        if num_current_patches > 0:
            image_type_ids[:num_current_patches] = 0
        history_traj_mask = self._build_history_traj_mask(data_item, num_tiles, num_patches)

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            loss_weight=ret['loss_weight'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            # pixel_labels=pixel_labels,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            image_type_ids=image_type_ids,
            history_traj_mask=history_traj_mask,
            # traversability_labels=traversability_labels
        )
        return ret


    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                logger.info(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        logger.info(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        data_path = os.path.join(self.root, data_item['image'])
                        logger.info(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                else:
                    raise NotImplementedError
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]



def build_datasets(
    data_args,
    tokenizer,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())

    # iterate all datasets in the meta file
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        # how many time the dataset is repeated
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style,
            ds_collections[ds_name],
            tokenizer,
            ds_name=ds_name,
            num_image_token=model.num_image_token, # 256
            image_size=data_args.force_image_size, # 448
            is_train=ds_collections[ds_name]['data_augment'], # use data_augment to indicate train or val (should be false when no data augment)
            pad2square=data_args.pad2square, # pad image to square
            group_by_length=group_by_length and not data_args.use_packed_ds, # when using packed ds, no need to group by length
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,  # how many time each sample is repeated
            normalize_type=normalize_type, # imagenet
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
            use_coordinate_loss_weight=data_args.use_coordinate_loss_weight,
            coordinate_loss_weights_with_space=data_args.coordinate_loss_weights_with_space,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)} ({data_rank=}, {data_world_size=})')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        raise NotImplementedError("Data resampling is not supported in lazy mode without packed dataset.")
    else:
        raise NotImplementedError("Only packed dataset is supported in lazy mode.")
    return train_dataset
