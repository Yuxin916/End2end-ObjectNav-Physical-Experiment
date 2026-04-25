from dataclasses import dataclass, field
from typing import Optional, Literal


"""
Training Parameters: https://huggingface.co/docs/transformers/v4.37.2/en/index

#### Saving & Logging Control ####
output_dir:
    where the wandb and training_log.txt and checkpoints will be saved
overwrite_output_dir: 
    If True, overwrite the content of the output directory. 
    If false, continue training if output_dir points to a checkpoint directory.
report_to:
    tensorboard, wandb,
save_strategy:
    "epoch": Save is done at the end of each epoch.
    "steps": Save is done every save_steps
    "no": No save is done during training.
save_total_limit:
    limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
    When load_best_model_at_end is enabled, the “best” checkpoint according to metric_for_best_model will 
    always be retained in addition to the most recent ones. 
save_steps:
    Number of updates steps before two checkpoint saves if save_strategy="steps". 
    Should be an integer or a float in range [0,1). 
    If smaller than 1, will be interpreted as ratio of total training steps
logging_strategy:
    "no": No logging is done during training.
    "epoch": Logging is done at the end of each epoch.
    "steps": Logging is done every logging_steps
logging_steps:
    Number of update steps between two logs if logging_strategy="steps". 
    Should be an integer or a float in range [0,1). 
    If smaller than 1, will be interpreted as ratio of total training steps.
evaluation_strategy:
    The evaluation strategy to adopt during training. 
    "no": No evaluation is done during training.
    "steps": Evaluation is done (and logged) every eval_steps.
    "epoch": Evaluation is done at the end of each epoch.


#### DataLoader Control ####
group_by_length： 
    Whether or not to group together samples of roughly the same length in the training 
    dataset (to minimize padding applied and be more efficient). 
    Only useful if applying dynamic padding. when data packing is used, no need to set this to True.
dataloader_num_workers 
    Number of subprocesses to use for data loading (PyTorch only). 
    0 means that the data will be loaded in the main process.


#### Training Precision Control ####
do_train:
    Whether to run training or not.
bf16:
    Whether to use bf16 (mixed) precision training.
learning_rate (float, optional, defaults: to 5e-5)
    initial learning rate for AdamW optimizer
weight_decay (float, optional, defaults to 0) 
    The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.
adam_beta1 (float, optional, defaults to 0.9) 
    The beta1 hyperparameter for the AdamW optimizer.
adam_beta2 (float, optional, defaults to 0.999) 
    The beta2 hyperparameter for the AdamW optimizer.
adam_epsilon (float, optional, defaults to 1e-8) 
    The epsilon hyperparameter for the AdamW optimizer.
max_grad_norm (float, optional, defaults to 1.0) 
    Maximum gradient norm (for gradient clipping).
lr_scheduler_type
    
warmup_ratio (float, optional, defaults to 0.0) 
    Ratio of total training steps used for a linear warmup from 0 to learning_rate
    
#### Training Steps Control ####
    max_steps:
        If > 0: set total number of training steps to perform. 
        Override num_train_epochs.
    num_train_epochs:
        Total number of training epochs to perform.
        An epoch is a single pass through the training dataset.
    per_device_train_batch_size
        The batch size per GPU/TPU core/CPU for training.
    world_size
        Number of distributed processes. (= number of GPUs if using distributed training)
    gradient_accumulation_steps
        Number of updates steps to accumulate before performing a backward/update pass.
    
    
    total_train_batch_size 每步的全局批量 = per_device_train_batch_size × world_size × gradient_accumulation_steps
    num_examples = len(train_dataset)
    
    if set num_train_epochs: [all data in train_dataset used for training]
        len_dataloader = ceil(num_examples / (per_device_train_batch_size × world_size))
        num_update_steps_per_epoch = max(len_dataloader // gradient_accumulation_steps, 1)
        max_steps = ceil(num_train_epochs × num_update_steps_per_epoch)
        
    else if set max_steps: [regardless of length of train_dataset]
        num_train_samples = max_steps × total_train_batch_size
        (num_examples remains the actual dataset length, but may not be fully used)

        
"""


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """

    """
    if     
    model_name_or_path set → Load entire model from there.
    else:
    model_name_or_path None → Use vision_path + llm_path (and optionally mlp_path) to assemble.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    """
    freeze_backbone	    unfreeze_vit_layers	    use_backbone_lora	    Result
        True	                0	                    0	        Vision fully frozen
        True	                2	                    0	        Vision frozen except last 2 blocks
        True	                0	                    16	        Vision frozen except LoRA modules rank 16
        False	                —	                    —	        Vision fully trainable

    freeze_llm	        unfreeze_lm_head	    use_llm_lora	        Result
        True	                False	                0	        LLM fully frozen
        True	                True	                0	        Only lm_head trains
        True	                False	                16	        LLM frozen except LoRA adapters rank 16
        False	                —	                    —	        LLM fully trainable
    """
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
        # adapting visual representations slightly while keeping the rest of the model frozen
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
        # last layer of the ViT -> most abstract representation
        # earlier layer -> finer, more localized features
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
        # final layer of the language model to be trainable, while the other parts may still be frozen
        # adapt the output distribution of the LLM to the downstream task
    )
    """
    grad_checkpoint: saves memory, costs speed. 
    drop_path_rate: randomly skip entire residual blocks in ViT, prevents overfitting in vision backbones
    """
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
        #  regularization for the Vision Transformer.
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    """
    data preprocessing parameters:
    1. max_seq_length: maximum total length of the input sequence after tokenization
                    text token + special tokens (not counting vision context tokens).
                    Longer gets truncated; shorter padded.
    2. force_image_size: The target resolution the image is resized to before patch embedding.
                    Habitat RGB images 640×480 ’ll be resized to 448×448 (with possible aspect distortion unless pad2square=True
    3. down_sample_ratio: Ratio by which the spatial resolution of the vision tokens is reduced after patch embedding.
                    eg: 0.5 means 32×32=1024 tokens → 16×16=256 tokens (per image)
    4. pad2square: Whether to pad the image to a square before resizing.
                    Prevents aspect ratio distortion
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        # roughly handle between 5400 to 6300 words
        # 640x480 image -> 1200 tokens
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    """
    Conversation style & dataset metadata
    1. conv_style: in conversation.py, register_conv_template
    2. meta_path: path to the metadata file of the dataset
    3. use_data_resampling: WeightedConcatDataset according to dataset lengths (how many data sources in meta_path)
    4. dynamic_image_size: randomly generate different number of patches for each image
        during training to improve generalization (similar to data augmentation)
    5. use_thumbnail: a small set of tokens that carry global context instead of patches
    6. min_dynamic_patch: minimum number of dynamic patches when dynamic_image_size is True
    7. max_dynamic_patch: maximum number of dynamic patches when dynamic_image_size is True
    """
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
        # strategy used to balance the distribution of classes in a dataset,
        # either by oversampling minority classes,
        # under-sampling majority classes, or generating synthetic data
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
        # resizing images differently on the fly to achieve better generalization.
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    # Dynamic patches are a more flexible approach where the size, number, or focus area of the patches can vary
    # based on the content of the image or the training context.
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    """
    Video support: Minimum/maximum frames per video sample.
    """
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    """
    Image normalization
    """
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    """
    Data Packing 
    1. use_packed_ds: Whether to use packed dataset for efficient training.
    
    """
    # packed dataset: multiple smaller data samples are packed together into a larger "super-sample"
    # input data may vary significantly in size (process sequential or variable-length inputs)
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    """
    Loss configuration
    """
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )
    """
    My custom model architecture modification
    """
    
    # numeric loss
    lambda_num: float = field(
        default=0.01,
        metadata={'help': 'Weight for auxiliary numeric-aware coordinate regression loss.'},
    )
    use_numeric_aux_loss: bool = field(
        default=False,
        metadata={'help': 'Whether to enable auxiliary numeric-aware coordinate regression loss during training.'},
    )
    
    # angle loss
    lambda_angle: float = field(
        default=0.0,
        metadata={'help': 'Weight for auxiliary angle loss computed from predicted/target coordinates.'},
    )
    use_angle_aux_loss: bool = field(
        default=False,
        metadata={'help': 'Whether to enable auxiliary angle loss during training.'},
    )
    angle_anchor_x: float = field(
        default=499.5,
        metadata={'help': 'Egocentric bottom-center anchor x coordinate for angle loss.'},
    )
    angle_anchor_y: float = field(
        default=999.0,
        metadata={'help': 'Egocentric bottom-center anchor y coordinate for angle loss.'},
    )
    
    # coordinate loss
    lambda_coord_dist: float = field(
        default=0.0,
        metadata={'help': 'Weight for grouped coordinate-distribution auxiliary loss.'},
    )
    use_coord_dist_aux_loss: bool = field(
        default=False,
        metadata={'help': 'Whether to enable grouped coordinate-distribution auxiliary loss during training.'},
    )
    coord_dist_sigma: float = field(
        default=15.0,
        metadata={'help': 'Gaussian sigma in [0,999] coordinate units for coordinate-distribution loss.'},
    )
    coord_dist_left_max: float = field(
        default=30.0,
        metadata={'help': 'Left action-region upper bound in [0,999] coordinate units for coordinate-distribution loss.'},
    )
    coord_dist_right_min: float = field(
        default=970.0,
        metadata={'help': 'Right action-region lower bound in [0,999] coordinate units for coordinate-distribution loss.'},
    )
    coord_dist_stop_min: float = field(
        default=940.0,
        metadata={'help': 'Stop action-region lower y bound in [0,999] coordinate units for coordinate-distribution loss.'},
    )
    
    # action-hinge loss
    lambda_action_hinge: float = field(
        default=0.05,
        metadata={'help': 'Weight for forward-aware action-region hinge auxiliary loss.'},
    )
    use_action_hinge_aux_loss: bool = field(
        default=False,
        metadata={'help': 'Whether to enable forward-aware action-region hinge auxiliary loss during training.'},
    )
    action_hinge_left_max: float = field(
        default=30.0,
        metadata={'help': 'Left action-region upper bound in [0,999] coordinate units for action-region hinge loss.'},
    )
    action_hinge_right_min: float = field(
        default=970.0,
        metadata={'help': 'Right action-region lower bound in [0,999] coordinate units for action-region hinge loss.'},
    )
    action_hinge_stop_min: float = field(
        default=940.0,
        metadata={'help': 'Stop action-region lower y bound in [0,999] coordinate units for action-region hinge loss.'},
    )
    action_hinge_forward_alpha_left: float = field(
        default=3.0,
        metadata={'help': 'Extra penalty multiplier for FORWARD samples predicted into left padding region.'},
    )
    action_hinge_forward_alpha_right: float = field(
        default=3.0,
        metadata={'help': 'Extra penalty multiplier for FORWARD samples predicted into right padding region.'},
    )
    action_hinge_forward_alpha_stop: float = field(
        default=3.0,
        metadata={'help': 'Extra penalty multiplier for FORWARD samples predicted into stop (bottom padding) region.'},
    )
   
    
    # loss weighting
    use_coordinate_loss_weight: bool = field(
        default=False,
        metadata={'help': 'Enable per-position coordinate loss weighting for assistant answers like XXX, YYY.'},
    )
    coordinate_loss_weights_with_space: Optional[str] = field(
        default='1,1,1,1,1,1,1,1',
        metadata={'help': 'Comma-separated weights for coordinate tokens. Length 8 = "XXX, YYY" only; length 9 = "XXX, YYY" plus EOS/im_end.'},
    )
    
    # special embeddings
    use_image_type_embeddings: bool = field(
        default=False,
        metadata={'help': 'Enable learnable current-vs-history visual type embeddings.'},
    )
    use_history_temporal_embeddings: bool = field(
        default=False,
        metadata={'help': 'Enable fixed sinusoidal temporal embeddings for history images.'},
    )
    use_padding_embeddings: bool = field(
        default=False,
        metadata={'help': 'Enable binary region embedding for visual tokens (rgb vs padding).'},
    )
    use_history_trajectory_embeddings: bool = field(
        default=False,
        metadata={'help': 'Enable binary embedding on history visual tokens (with trajectory vs without trajectory).'},
    )
