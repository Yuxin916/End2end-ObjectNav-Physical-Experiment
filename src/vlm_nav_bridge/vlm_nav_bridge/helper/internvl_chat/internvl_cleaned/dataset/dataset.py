from torch.utils.data import Dataset
import torch.distributed as dist

import json
import os
import math
import random
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from copy import deepcopy
import traceback
from typing import Dict, Literal, Optional

from internvl_cleaned.dataset.data_preprocess import (build_transform,
                                                      dynamic_preprocess,
                                                      preprocess_internvl2_5
                                                      )
from internvl_cleaned.dataset.data_packing import PackedDataset
from internvl_cleaned.constants_utils import IMG_END_TOKEN
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _compute_angular_soft_weights(agent_pos, candidates, selected_id, sigma_deg=25.0):
    """Von Mises soft labels based on bearing from agent to each candidate.

    Returns None if any candidate is 'target' (use hard CE for goal-reaching steps
    to prevent goal-dithering from soft supervision).
    """
    if any(c.get('type') == 'target' for c in candidates):
        return None
    agent_r, agent_c = agent_pos[0], agent_pos[1]
    gt_cand = next((c for c in candidates if c['id'] == selected_id), None)
    if gt_cand is None:
        return None
    gr, gc = gt_cand['pos'][0], gt_cand['pos'][1]
    theta_gt = math.atan2(-(gr - agent_r), gc - agent_c)  # row increases down → negate dy
    kappa = (math.radians(sigma_deg)) ** -2
    weights = []
    for cand in candidates:
        cr, cc = cand['pos'][0], cand['pos'][1]
        theta_i = math.atan2(-(cr - agent_r), cc - agent_c)
        delta = math.atan2(math.sin(theta_i - theta_gt), math.cos(theta_i - theta_gt))  # wrap to [-π,π]
        weights.append(math.exp(kappa * (math.cos(delta) - 1)))  # -1 so max=1 at delta=0
    total = sum(weights) + 1e-8
    return [w / total for w in weights]


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
        num_image_token_bev=None,
        num_image_token_ego=None,
        use_dual_vit=False,
        use_angular_soft_ce=False,
        angular_sigma_deg=25.0,
    ):
        # Initialize the dataset class
        super(LazySupervisedDataset, self).__init__()

        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.num_image_token_bev = num_image_token_bev
        self.num_image_token_ego = num_image_token_ego
        self.use_dual_vit = bool(use_dual_vit)
        self.use_angular_soft_ce = bool(use_angular_soft_ce)
        self.angular_sigma_deg = float(angular_sigma_deg)

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

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            raise NotImplementedError("group_by_length is not supported in lazy mode.")

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'internvl2_5' or self.template_name == 'internvl2_5_nav':
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

    def multi_modal_get_item(self, data_item):
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
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Extract position information if available
        position_info = None
        if 'position_info' in data_item:
            position_info = data_item['position_info']

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )

        # Add position information if available
        if position_info is not None:
            ret['position_info'] = position_info
        # Keep packed samples schema-consistent with multi-image path.
        if self.use_packed_ds:
            if self.use_dual_vit and self.num_image_token_bev is not None:
                image_token_count = self.num_image_token_bev
            else:
                image_token_count = self.num_image_token * num_patches
            ret['image_token_counts'] = torch.tensor([image_token_count], dtype=torch.long)

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

        # import torch
        # import matplotlib.pyplot as plt
        # from torchvision.transforms.functional import to_pil_image
        #
        # # denormalize it and view it
        # MEAN = (0.485, 0.456, 0.406)
        # STD = (0.229, 0.224, 0.225)
        #
        # def denormalize(tensor, mean, std):
        #     """Undo normalization."""
        #     for t, m, s in zip(tensor, mean, std):
        #         t.mul_(s).add_(m)
        #     return tensor.clamp(0, 1)
        #
        # # visualize all 3
        # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # for i, tensor in enumerate(pixel_values):
        #     img = denormalize(tensor.clone(), MEAN, STD)
        #     img = to_pil_image(img)
        #     axes[i].imshow(img)
        #     axes[i].axis("off")
        #     axes[i].set_title(f"Image {i + 1}")
        # plt.tight_layout()
        # plt.show()
        #
        #
        #

        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Dual-ViT: first image BEV (num_image_token_bev), rest ego (num_image_token_ego each)
        is_dual_vit = self.use_dual_vit
        if (is_dual_vit and self.num_image_token_bev is not None
                and self.num_image_token_ego is not None):
            num_image_tokens = [self.num_image_token_bev] + [self.num_image_token_ego] * (num_image - 1)
        else:
            num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Extract position information if available
        position_info = None
        if 'position_info' in data_item:
            position_info = data_item['position_info']

        # PosD: compute angular soft CE weights on-the-fly
        if (self.use_angular_soft_ce
                and position_info is not None
                and 'candidates' in position_info
                and 'selected_candidate_id' in position_info):
            weights = _compute_angular_soft_weights(
                position_info['agent_pos'],
                position_info['candidates'],
                position_info['selected_candidate_id'],
                sigma_deg=self.angular_sigma_deg,
            )
            if weights is not None:
                cand_ids = [c['id'] for c in position_info['candidates']]
                position_info = dict(position_info)  # shallow copy to avoid mutating shared data
                position_info['soft_label_weights'] = weights
                position_info['soft_label_candidate_ids'] = cand_ids

        # Create the final return dictionary
        out = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        if position_info is not None:
            out['position_info'] = position_info
        # Always emit per-image token counts for packed consistency. Dual-ViT consumes it explicitly.
        if self.use_packed_ds:
            out['image_token_counts'] = torch.tensor(num_image_tokens, dtype=torch.long)
        return out


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
    inferred_dual_vit = (
        getattr(data_args, 'num_image_token_bev', None) is not None
        and getattr(data_args, 'num_image_token_ego', None) is not None
    )
    use_dual_vit = data_args.use_dual_vit if getattr(data_args, 'use_dual_vit', None) is not None else inferred_dual_vit

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
            num_image_token_bev=getattr(data_args, 'num_image_token_bev', None),
            num_image_token_ego=getattr(data_args, 'num_image_token_ego', None),
            use_dual_vit=use_dual_vit,
            use_angular_soft_ce=getattr(data_args, 'use_angular_soft_ce', False),
            angular_sigma_deg=getattr(data_args, 'angular_sigma_deg', 25.0),
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