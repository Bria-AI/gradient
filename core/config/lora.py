from typing import Optional, Tuple
from pydantic import BaseModel, Field, model_validator, field_validator


class LoraConfig(BaseModel):
    pretrained_model_name_or_path: str = Field(
        default="briaai/BRIA-4B-Adapt",
        description="Pretrained model name or path from Hugging Face model hub.",
    )
    revision: Optional[str] = Field(
        default=None,
        description="Revision of pretrained model identifier from Hugging Face hub.",
    )
    variant: Optional[str] = Field(
        default=None, description="Variant of the model files (e.g., fp16)."
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Name of the dataset on Hugging Face hub containing training data.",
    )
    dataset_config_name: Optional[str] = Field(
        default=None, description="Configuration name of the dataset, if applicable."
    )
    instance_data_dir: Optional[str] = Field(
        default=None, description="Path to the directory containing training data."
    )
    cache_dir: Optional[str] = Field(
        default=None, description="Directory to cache downloaded models and datasets."
    )
    image_column: str = Field(
        default="image", description="Column name containing image data."
    )
    caption_column: str = Field(
        default="caption", description="Column name containing captions for each image."
    )
    repeats: int = Field(
        default=1, description="Number of times to repeat the training data."
    )
    class_data_dir: Optional[str] = Field(
        default=None, description="Path to the directory containing class images."
    )
    instance_prompt: Optional[str] = Field(
        default="None",
        description="Prompt specifying the instance (e.g., 'photo of a TOK dog').",
    )
    class_prompt: Optional[str] = Field(
        default=None, description="Prompt specifying the class of images."
    )
    max_sequence_length: int = Field(
        default=128, description="Maximum sequence length for the text encoder."
    )
    rank: int = Field(default=128, description="Dimension of the LoRA update matrices.")
    with_prior_preservation: bool = Field(
        default=False, description="Flag to enable prior preservation loss."
    )
    prior_loss_weight: float = Field(
        default=1.0, description="Weight for prior preservation loss."
    )
    num_class_images: int = Field(
        default=100,
        description="Minimum number of class images for prior preservation loss.",
    )
    output_dir: str = Field(
        default="bria-dreambooth-lora",
        description="Directory to save the model predictions and checkpoints.",
    )
    seed: Optional[int] = Field(
        default=None, description="Seed for reproducible training."
    )
    resolution: int = Field(default=1024, description="Resolution for input images.")
    center_crop: bool = Field(
        default=True,
        description="Center crop input images to the specified resolution.",
    )
    random_flip: bool = Field(
        default=False, description="Randomly flip images horizontally."
    )
    train_batch_size: int = Field(default=1, description="Batch size for training.")
    sample_batch_size: int = Field(
        default=4, description="Batch size for sampling images."
    )
    num_train_epochs: int = Field(default=1, description="Number of training epochs.")
    max_train_steps: Optional[int] = Field(
        default=None,
        description="Total number of training steps. Overrides num_train_epochs if provided.",
    )
    checkpointing_steps: int = Field(
        default=250, description="Save a checkpoint every X updates."
    )
    checkpoints_total_limit: Optional[int] = Field(
        default=None, description="Maximum number of checkpoints to store."
    )
    resume_from_checkpoint: Optional[str] = Field(
        default=None, description="Path to a checkpoint to resume training from."
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        description="Number of steps to accumulate gradients before an update.",
    )
    gradient_checkpointing: bool = Field(
        default=False, description="Enable gradient checkpointing to save memory."
    )
    learning_rate: float = Field(default=1.0, description="Initial learning rate.")
    guidance_scale: float = Field(
        default=1.0, description="Guidance scale for models like FLUX.1 dev variant."
    )
    scale_lr: bool = Field(
        default=False,
        description="Scale learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    lr_scheduler: str = Field(
        default="constant", description="Learning rate scheduler type."
    )
    lr_warmup_steps: int = Field(
        default=0, description="Number of steps for learning rate warmup."
    )
    lr_num_cycles: int = Field(
        default=1, description="Number of LR resets in cosine_with_restarts scheduler."
    )
    lr_power: float = Field(
        default=1.0, description="Power factor for polynomial scheduler."
    )
    dataloader_num_workers: int = Field(
        default=0, description="Number of workers for data loading."
    )
    weighting_scheme: str = Field(
        default="none",
        description="Weighting scheme for sampling and loss.",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    logit_mean: float = Field(
        default=0.0, description="Mean for logit_normal weighting scheme."
    )
    logit_std: float = Field(
        default=1.0, description="Standard deviation for logit_normal weighting scheme."
    )
    mode_scale: float = Field(
        default=1.29, description="Scale for mode weighting scheme."
    )
    optimizer: str = Field(
        default="prodigy", description="Optimizer type.", choices=["AdamW", "prodigy"]
    )
    use_8bit_adam: bool = Field(default=False, description="Use 8-bit Adam optimizer.")
    adam_beta1: float = Field(
        default=0.9, description="Beta1 parameter for Adam optimizer."
    )
    adam_beta2: float = Field(
        default=0.999, description="Beta2 parameter for Adam optimizer."
    )
    prodigy_beta3: Optional[float] = Field(
        default=None, description="Beta3 parameter for Prodigy optimizer."
    )
    prodigy_decouple: bool = Field(
        default=True, description="Use decoupled weight decay for Prodigy optimizer."
    )
    adam_weight_decay: float = Field(
        default=1e-4, description="Weight decay for U-Net parameters."
    )
    adam_weight_decay_text_encoder: float = Field(
        default=1e-3, description="Weight decay for text encoder."
    )
    adam_epsilon: float = Field(
        default=1e-08, description="Epsilon for Adam optimizer."
    )
    prodigy_use_bias_correction: bool = Field(
        default=True, description="Use bias correction for Prodigy optimizer."
    )
    prodigy_safeguard_warmup: bool = Field(
        default=True, description="Safeguard warmup for Prodigy optimizer."
    )
    max_grad_norm: float = Field(
        default=1.0, description="Maximum gradient norm for clipping."
    )
    logging_dir: str = Field(
        default="logs", description="Directory for logging (e.g., TensorBoard)."
    )
    allow_tf32: bool = Field(
        default=False,
        description="Allow TensorFloat32 on Ampere GPUs for faster training.",
    )
    cache_latents: bool = Field(
        default=False, description="Cache VAE latents for efficiency."
    )
    report_to: str = Field(
        default="tensorboard",
        description="Integration for reporting logs and results.",
        choices=["tensorboard", "wandb", "comet_ml", "all"],
    )
    mixed_precision: str = Field(
        default="bf16",
        description="Use mixed precision training.",
        choices=["no", "fp16", "bf16"],
    )
    upcast_before_saving: bool = Field(
        default=False, description="Upcast trained layers to float32 before saving."
    )
    prior_generation_precision: Optional[str] = Field(
        default=None,
        description="Precision for prior generation.",
        choices=["no", "fp32", "fp16", "bf16"],
    )
    local_rank: int = Field(
        default=-1, description="Local rank for distributed training."
    )

    @model_validator(mode="after")
    def validate_dataset_or_instance_dir(cls, values):
        dataset_name = values.dataset_name
        instance_data_dir = values.instance_data_dir
        if not dataset_name and not instance_data_dir:
            raise ValueError("Specify either `dataset_name` or `instance_data_dir`.")
        if dataset_name and instance_data_dir:
            raise ValueError(
                "Specify only one of `dataset_name` or `instance_data_dir`."
            )
        return values

    @field_validator("class_data_dir")
    def validate_class_data_dir(cls, value, info):
        if info.data.get("with_prior_preservation") and value is None:
            raise ValueError(
                "Specify `class_data_dir` when `with_prior_preservation` is enabled."
            )
        return value

    @field_validator("class_prompt")
    def validate_class_prompt(cls, value, info):
        if info.data.get("with_prior_preservation") and value is None:
            raise ValueError(
                "Specify `class_prompt` when `with_prior_preservation` is enabled."
            )
        return value
