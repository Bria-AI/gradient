import os
from typing import List, Optional

from pydantic import BaseModel, Field


class HyperParemeter(BaseModel):
    pretrained_vae_model_name_or_path: str = Field(
        default="black-forest-labs/FLUX.1-schnell",
        description="Path to an improved VAE to stabilize training. For more details, see https://github.com/huggingface/diffusers/pull/4038.",
    )
    pretrained_text_encoder_name_or_path: str = Field(
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        description="Using it's text encoder google/t5-v1_1-xxl.",
    )
    revision: Optional[str] = Field(
        default=None,
        description="Revision of pretrained model identifier from huggingface.co/models.",
    )
    # TODO: checkpoint saving
    s3_bucket_name: str = Field(
        default="your-s3-bucket",
        description="S3 bucket for saving checkpoints.",
    )
    s3_prefix: str = Field(
        default="your-s3-prefix",
        description="S3 directory saving checkpoints.",
    )

    max_train_samples: Optional[int] = Field(
        default=None,
        description="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    )
    seed: int = Field(
        default=10,
        description="A seed for reproducible training.",
    )
    save_metrics: bool = Field(
        default=True,
        description="Save metrics during training.",
    )
    resolution: int = Field(
        default=256,
        description="The resolution for input images.",
    )
    center_crop: int = Field(
        default=0,
        description="Whether to center crop the input images to the resolution.",
    )
    h_flip: int = Field(
        default=0,
        description="Whether to horizontally flip the input images randomly.",
    )
    resize: int = Field(
        default=0,
        description="Resize input images to a specific resolution.",
    )
    max_sequence_length: int = Field(
        default=128,
        description="Maximum sequence length for the T5 text encoder.",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    learning_rate: float = Field(
        default=1e-4,
        description="Initial learning rate to use.",
    )
    lr_scheduler: str = Field(
        default="constant_with_warmup",
        description="The scheduler type to use.",
    )
    lr_warmup_steps: int = Field(
        default=10000,
        description="Number of steps for the warmup in the LR scheduler.",
    )
    allow_tf32: bool = Field(
        default=True,
        description="Whether or not to allow TF32 on Ampere GPUs.",
    )
    weighting_scheme: str = Field(
        default="logit_normal",
        description="Weighting scheme for training.",
    )
    logit_mean: float = Field(
        default=0.0,
        description="Mean to use when using the 'logit_normal' weighting scheme.",
    )
    logit_std: float = Field(
        default=1.0,
        description="Std to use when using the 'logit_normal' weighting scheme.",
    )
    mode_scale: float = Field(
        default=1.29,
        description="Scale of mode weighting scheme.",
    )
    # TODO: this is optimizer config?
    ############################
    use_8bit_adam: bool = Field(
        default=False,
        description="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    use_adafactor: int = Field(
        default=0,
        description="Use AdaFactor optimizer.",
    )
    adam_beta1: float = Field(
        default=0.9,
        description="The beta1 parameter for the Adam optimizer.",
    )
    adam_beta2: float = Field(
        default=0.999,
        description="The beta2 parameter for the Adam optimizer.",
    )
    adam_weight_decay: float = Field(
        default=1e-4,
        description="Weight decay to use.",
    )
    adam_epsilon: float = Field(
        default=1e-08,
        description="Epsilon value for the Adam optimizer.",
    )
    max_grad_norm: float = Field(
        default=1.0,
        description="Max gradient norm.",
    )
    mixed_precision: str = Field(
        default="bf16",
        description="Whether to use mixed precision.",
    )
    noise_offset: float = Field(
        default=0.0,
        description="The scale of noise offset.",
    )

    local_rank: int = Field(
        default=-1,
        description="For distributed training: local_rank.",
    )

    save_pipeline: bool = Field(
        default=False,
        description="Whether to save only the pipeline instead of the entire accelerator.",
    )
    no_cfg: bool = Field(
        default=False,
        description="Avoid replacing 10% of captions with null embeddings.",
    )
    drop_rate_cfg: float = Field(
        default=0.1,
        description="Rate for Classifier Free Guidance dropping.",
    )
    dense_caption_ratio: float = Field(
        default=0.5,
        description="Rate for dense captions.",
    )

    enable_xformers_memory_efficient_attention: bool = Field(
        default=False,
        description="Whether or not to use xformers.",
    )

    first_ema_step: bool = Field(
        default=False,
        description="Initialize EMA model according to the Unet.",
    )
    crops_coords_top_left_h: int = Field(
        default=0,
        description="Coordinate for height to be included in the crop coordinate embeddings needed by SDXL Unet.",
    )
    crops_coords_top_left_w: int = Field(
        default=0,
        description="Coordinate for width to be included in the crop coordinate embeddings needed by SDXL Unet.",
    )
    convert_unet_to_weight_dtype: bool = Field(
        default=False,
        description="Convert Unet to weight_dtype.",
    )
    precompute: bool = Field(
        default=False,
        description="Use precomputed latents and text embeddings.",
    )
    random_latents: int = Field(
        default=0,
        description="Use precomputed latents and text embeddings.",
    )
    reinit_scheduler: int = Field(
        default=0,
        description="Reinitialize the scheduler.",
    )
    reinit_optimizer: int = Field(
        default=0,
        description="Reinitialize the optimizer.",
    )
    reinit_optimizer_type: str = Field(
        default="",
        description="Type of optimizer to reinitialize.",
    )

    train_with_ratios: int = Field(
        default=1,
        description="Train using ratios.",
    )
    # TODO: remove this?
    curated_training: Optional[str] = Field(
        default=None,
        description="Use curated training mode for bucketing.",
    )
    debug: int = Field(
        default=1,
        description="Debug mode flag.",
    )
    gradient_checkpointing: int = Field(
        default=0,
        description="Enable gradient checkpointing.",
    )
    force_download: bool = Field(
        default=True,
        description="Force download from hub.",
    )
    low_res_fine_tune: int = Field(
        default=0,
        description="Low resolution fine-tuning.",
    )
    shift: float = Field(
        default=1.0,
        description="Noise shifting.",
    )
    variant: Optional[str] = Field(
        default=None,
        description="Variant of model files for the pretrained model.",
    )
    use_flow_matching: int = Field(
        default=1,
        description="Use flow matching.",
    )
    compile: int = Field(
        default=0,
        description="Compile transformer.",
    )
    flow_matching_latent_loss: int = Field(
        default=0,
        description="Use latent loss instead of model pred loss.",
    )
    use_continuous_sigmas: int = Field(
        default=0,
        description="Enable continuous sigmas.",
    )
    rope_theta: int = Field(
        default=10000,
        description="Rope frequency.",
    )
    time_theta: int = Field(
        default=10000,
        description="Time embed frequency.",
    )
    use_dynamic_shift: int = Field(
        default=0,
        description="Enable dynamic shift.",
    )

    def get_hyperparameter(self) -> dict:
        args = self.model_dump()

        print(f"flow_matching_latent_loss: {args.flow_matching_latent_loss}")

        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != args.local_rank:
            args.local_rank = env_local_rank

        # Sanity checks
        if args.dataset_name is None and args.data_channels == "":
            raise ValueError("Need either a dataset name or a training folder.")

        assert (
            args.s3_prefix is not None
        ), "s3_prefix must be specified, i.e., dir for saving checkpoints at s3"

        # Init boolean args that are ints
        args.reinit_scheduler = args.reinit_scheduler == 1
        args.use_adafactor = args.use_adafactor == 1
        args.reinit_optimizer = args.reinit_optimizer == 1
        args.train_with_ratios == (args.train_with_ratios == 1)

        return args
