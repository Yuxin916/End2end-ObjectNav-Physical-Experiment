# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class InternVLChatConfig(PretrainedConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(
            self,
            vision_config=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            ps_version='v1',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            use_image_type_embeddings=False,
            use_history_temporal_embeddings=False,
            use_padding_embeddings=False,
            use_padding_embeeding=None,
            use_history_trajectory_embeddings=False,
            use_coord_dist_aux_loss=False,
            lambda_coord_dist=0.0,
            coord_dist_sigma=15.0,
            coord_dist_left_max=30.0,
            coord_dist_right_min=970.0,
            coord_dist_stop_min=940.0,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'architectures': ['InternVisionModel']}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if llm_config is None:
            # TODO: There might still be a bug in transformers version 4.44 and above.
            llm_config = {'architectures': ['']}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

        # config the vision encoder
        # configuration_intern_vit.py
        self.vision_config = InternVisionConfig(**vision_config)

        # config the language model
        if llm_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        else:
            raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_image_type_embeddings = use_image_type_embeddings
        self.use_history_temporal_embeddings = use_history_temporal_embeddings
        # Backward compatibility: allow legacy typo key while canonicalizing to use_padding_embeddings.
        if use_padding_embeeding is not None:
            use_padding_embeddings = use_padding_embeeding
        self.use_padding_embeddings = use_padding_embeddings
        self.use_padding_embeeding = use_padding_embeddings
        self.use_history_trajectory_embeddings = use_history_trajectory_embeddings
        self.use_coord_dist_aux_loss = use_coord_dist_aux_loss
        self.lambda_coord_dist = lambda_coord_dist
        self.coord_dist_sigma = coord_dist_sigma
        self.coord_dist_left_max = coord_dist_left_max
        self.coord_dist_right_min = coord_dist_right_min
        self.coord_dist_stop_min = coord_dist_stop_min

        self.hidden_size = self.llm_config.hidden_size
        # By default, we use tie_word_embeddings=False for models of all sizes.
        self.tie_word_embeddings = False
        self.llm_config.tie_word_embeddings = self.tie_word_embeddings


    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch
        output['use_image_type_embeddings'] = self.use_image_type_embeddings
        output['use_history_temporal_embeddings'] = self.use_history_temporal_embeddings
        output['use_padding_embeddings'] = self.use_padding_embeddings
        output['use_padding_embeeding'] = self.use_padding_embeddings
        output['use_history_trajectory_embeddings'] = self.use_history_trajectory_embeddings
        output['use_coord_dist_aux_loss'] = self.use_coord_dist_aux_loss
        output['lambda_coord_dist'] = self.lambda_coord_dist
        output['coord_dist_sigma'] = self.coord_dist_sigma
        output['coord_dist_left_max'] = self.coord_dist_left_max
        output['coord_dist_right_min'] = self.coord_dist_right_min
        output['coord_dist_stop_min'] = self.coord_dist_stop_min

        return output
    
