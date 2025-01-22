from typing import Optional
from models.diffusion.bria4B_adapt.model_utils.pipeline_bria import BriaPipeline
from models.diffusion.bria4B_adapt.model_utils.transformer_bria import (
    BriaTransformer2DModel,
)
import logging
import math
import os
import copy
import shutil
from pathlib import Path
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import PretrainedConfig, T5TokenizerFast
from transformers import T5TokenizerFast, T5EncoderModel

from core.config.dataloader import DataLoaderConfig
from core.config.loggers import Logger
from core.config.optimizers import OptimizerParams as OptimizerConfig
from core.config.startegy import StrategyConfig
from core.config.trainer import TrainerConfig
from core.config.lora import LoraConfig
from core.config.dataset import DatasetConfig
from core.config.dream_booth_dataset import DreamBoothDataset

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
import torch.nn.functional as F
from prompt_dataset import PromptDataset, encode_prompt

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


class Bria4BAdaptLora:
    def collate_fn(self, examples, with_prior_preservation=False):
        pixel_values = [example["instance_images"] for example in examples]
        prompts = [example["instance_prompt"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior_preservation:
            pixel_values += [example["class_images"] for example in examples]
            prompts += [example["class_prompt"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        batch = {"pixel_values": pixel_values, "prompts": prompts}
        return batch

    def __init__(self, lora_config: LoraConfig):
        self.lora_config = lora_config

    def train(
        self,
        trainer_config: TrainerConfig,
        startegy_config: StrategyConfig,
        logger_config: Optional[Logger] = None,
    ):
        lora_config = self.lora_config

        if torch.backends.mps.is_available() and lora_config.mixed_precision == "bf16":
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        accelerator_project_config = ProjectConfiguration(
            project_dir=trainer_config.output_dir, logging_dir=logger_config.logging_dir
        )
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda"

        accelerator = Accelerator(
            gradient_accumulation_steps=lora_config.gradient_accumulation_steps,
            mixed_precision=lora_config.mixed_precision,
            log_with=logger_config.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )
        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        if logger_config.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError(
                    "Make sure to install wandb if you want to use it for logging during training."
                )

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if lora_config.seed is not None:
            set_seed(lora_config.seed)

        # Generate class images if prior preservation is enabled.
        if self.lora_config.with_prior_preservation:
            class_images_dir = Path(lora_config.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < lora_config.num_class_images:
                has_supported_fp16_accelerator = (
                    torch.cuda.is_available() or torch.backends.mps.is_available()
                )
                torch_dtype = (
                    torch.float16 if has_supported_fp16_accelerator else torch.float32
                )
                if lora_config.prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif lora_config.prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif lora_config.prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                pipeline = BriaPipeline.from_pretrained(
                    lora_config.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    revision=lora_config.revision,
                    variant=lora_config.variant,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = lora_config.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(lora_config.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(
                    sample_dataset, batch_size=lora_config.sample_batch_size
                )

                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)

                for example in tqdm(
                    sample_dataloader,
                    desc="Generating class images",
                    disable=not accelerator.is_local_main_process,
                ):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = (
                            class_images_dir
                            / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Handle the repository creation
        if accelerator.is_main_process:
            if trainer_config.output_dir is not None:
                os.makedirs(trainer_config.output_dir, exist_ok=True)

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            lora_config.pretrained_model_name_or_path,
            subfolder="scheduler",
            shift=4,
            use_dynamic_shifting=False,
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)

        # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        transformer = BriaTransformer2DModel.from_pretrained(
            lora_config.pretrained_model_name_or_path, subfolder="transformer"
        )
        transformer.to(accelerator.device, dtype=weight_dtype)

        tokenizer = T5TokenizerFast.from_pretrained(
            lora_config.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = T5EncoderModel.from_pretrained(
            lora_config.pretrained_model_name_or_path, subfolder="text_encoder"
        )

        # T5 is senstive to precision so we use the precision used for precompute and cast as needed
        T5_PRECISION = torch.float16
        text_encoder = text_encoder.to(dtype=T5_PRECISION)
        for block in text_encoder.encoder.block:
            block.layer[-1].DenseReluDense.wo.to(dtype=torch.float32)

        vae = AutoencoderKL.from_pretrained(
            lora_config.pretrained_model_name_or_path, subfolder="vae"
        )

        # We only train the additional adapter LoRA layers
        transformer.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device)  # do not change dtype, keep in T5_PRECISION

        if lora_config.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()

        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        # now we will add new LoRA weights to the attention layers
        transformer_lora_config = LoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(transformer_lora_config)

        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                transformer_lora_layers_to_save = None
                text_encoder_lora_layers_to_save = None

                for model in models:
                    if isinstance(model, type(unwrap_model(transformer))):
                        transformer_lora_layers_to_save = get_peft_model_state_dict(
                            model
                        )
                    elif isinstance(model, type(unwrap_model(text_encoder))):
                        text_encoder_lora_layers_to_save = get_peft_model_state_dict(
                            model
                        )
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

                BriaPipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            transformer_ = None
            text_encoder_ = None

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_ = model
                elif isinstance(model, type(unwrap_model(text_encoder_))):
                    text_encoder_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            lora_state_dict = BriaPipeline.lora_state_dict(input_dir)

            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(
                transformer_state_dict
            )
            incompatible_keys = set_peft_model_state_dict(
                transformer_, transformer_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if lora_config.mixed_precision == "fp16":
                models = [transformer_]
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if lora_config.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        if lora_config.scale_lr:
            lora_config.learning_rate = (
                lora_config.learning_rate
                * lora_config.gradient_accumulation_steps
                * lora_config.train_batch_size
                * accelerator.num_processes
            )

        # Make sure the trainable params are in float32.
        if lora_config.mixed_precision == "fp16":
            models = [transformer]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

        transformer_lora_parameters = list(
            filter(lambda p: p.requires_grad, transformer.parameters())
        )

        # Optimization parameters
        transformer_parameters_with_lr = {
            "params": transformer_lora_parameters,
            "lr": lora_config.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]

        # Optimizer creation
        if not (
            lora_config.optimizer.lower() == "prodigy"
            or lora_config.optimizer.lower() == "adamw"
        ):
            logger.warning(
                f"Unsupported choice of optimizer: {lora_config.optimizer}.Supported optimizers include [adamW, prodigy]."
                "Defaulting to adamW"
            )
            lora_config.optimizer = "adamw"

        if lora_config.use_8bit_adam and not lora_config.optimizer.lower() == "adamw":
            logger.warning(
                f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
                f"set to {lora_config.optimizer.lower()}"
            )

        if lora_config.optimizer.lower() == "adamw":
            if lora_config.use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    raise ImportError(
                        "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                    )

                optimizer_class = bnb.optim.AdamW8bit
            else:
                optimizer_class = torch.optim.AdamW

            optimizer = optimizer_class(
                params_to_optimize,
                betas=(lora_config.adam_beta1, lora_config.adam_beta2),
                weight_decay=lora_config.adam_weight_decay,
                eps=lora_config.adam_epsilon,
            )

        if lora_config.optimizer.lower() == "prodigy":
            try:
                import prodigyopt
            except ImportError:
                raise ImportError(
                    "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
                )

            optimizer_class = prodigyopt.Prodigy

            if lora_config.learning_rate <= 0.1:
                logger.warning(
                    "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
                )

            optimizer = optimizer_class(
                params_to_optimize,
                lr=lora_config.learning_rate,
                betas=(lora_config.adam_beta1, lora_config.adam_beta2),
                beta3=lora_config.prodigy_beta3,
                weight_decay=lora_config.adam_weight_decay,
                eps=lora_config.adam_epsilon,
                decouple=lora_config.prodigy_decouple,
                use_bias_correction=lora_config.prodigy_use_bias_correction,
                safeguard_warmup=lora_config.prodigy_safeguard_warmup,
            )

        # TODO: get as a parameter
        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            dataset_config=lora_config.dataset,
            instance_data_root=lora_config.instance_data_dir,
            instance_prompt=lora_config.instance_prompt,
            class_prompt=lora_config.class_prompt,
            class_data_root=lora_config.class_data_dir
            if lora_config.with_prior_preservation
            else None,
            class_num=lora_config.num_class_images,
            size=lora_config.resolution,
            repeats=lora_config.repeats,
            center_crop=lora_config.center_crop,
        )
        # TODO: get as a parameter
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=lora_config.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: self.collate_fn(
                examples, lora_config.with_prior_preservation
            ),
            num_workers=lora_config.dataloader_num_workers,
        )

        tokenizers = [tokenizer]
        text_encoders = [text_encoder]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, lora_config.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                # pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

        # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
        # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
        # the redundant encoding.
        # TODO: check what is instance_prompt
        if not train_dataset.custom_instance_prompts:
            (
                instance_prompt_hidden_states,
                instance_pooled_prompt_embeds,
                instance_text_ids,
            ) = compute_text_embeddings(
                lora_config.instance_prompt, text_encoders, tokenizers
            )

        # Handle class prompt for prior-preservation.
        if lora_config.with_prior_preservation:
            (
                class_prompt_hidden_states,
                class_pooled_prompt_embeds,
                class_text_ids,
            ) = compute_text_embeddings(
                lora_config.class_prompt, text_encoders, tokenizers
            )

        # Clear the memory here
        if not train_dataset.custom_instance_prompts:
            del text_encoder, tokenizer

        # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
        # pack the statically computed variables appropriately here. This is so that we don't
        # have to pass them to the dataloader.

        if not train_dataset.custom_instance_prompts:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
            if lora_config.with_prior_preservation:
                prompt_embeds = torch.cat(
                    [prompt_embeds, class_prompt_hidden_states], dim=0
                )
                pooled_prompt_embeds = torch.cat(
                    [pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0
                )
                text_ids = torch.cat([text_ids, class_text_ids], dim=0)

        vae_config_shift_factor = vae.config.shift_factor
        vae_config_scaling_factor = vae.config.scaling_factor
        vae_config_block_out_channels = vae.config.block_out_channels
        if lora_config.cache_latents:
            latents_cache = []
            for batch in tqdm(train_dataloader, desc="Caching latents"):
                with torch.no_grad():
                    batch["pixel_values"] = batch["pixel_values"].to(
                        accelerator.device, non_blocking=True, dtype=weight_dtype
                    )
                    latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

            del vae

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / lora_config.gradient_accumulation_steps
        )
        if trainer_config.max_train_steps is None:
            trainer_config.max_train_steps = (
                trainer_config.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            lora_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=lora_config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=lora_config.max_train_steps * accelerator.num_processes,
            num_cycles=lora_config.lr_num_cycles,
            power=lora_config.lr_power,
        )

        # Prepare everything with our `accelerator`.
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / lora_config.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            lora_config.max_train_steps = (
                trainer_config.num_train_epochs * num_update_steps_per_epoch
            )
        # Afterwards we recalculate our number of training epochs
        trainer_config.num_train_epochs = math.ceil(
            lora_config.max_train_steps / num_update_steps_per_epoch
        )

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            tracker_name = "dreambooth-bria-dev-lora"
            accelerator.init_trackers(tracker_name, config=vars(lora_config))

        # Train!
        total_batch_size = (
            lora_config.train_batch_size
            * accelerator.num_processes
            * lora_config.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {trainer_config.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {lora_config.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {lora_config.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {lora_config.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if trainer_config.resume_from_checkpoint:
            if trainer_config.resume_from_checkpoint != "latest":
                path = os.path.basename(trainer_config.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(trainer_config.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{trainer_config.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                trainer_config.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(trainer_config.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch

        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, lora_config.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = noise_scheduler_copy.sigmas.to(
                device=accelerator.device, dtype=dtype
            )
            schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)
            step_indices = [
                (schedule_timesteps == t).nonzero().item() for t in timesteps
            ]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        for epoch in range(first_epoch, trainer_config.num_train_epochs):
            transformer.train()

            for step, batch in enumerate(train_dataloader):
                models_to_accumulate = [transformer]
                with accelerator.accumulate(models_to_accumulate):
                    prompts = batch["prompts"]

                    # encode batch prompts when custom prompts are provided for each image -
                    if train_dataset.custom_instance_prompts:
                        (
                            prompt_embeds,
                            pooled_prompt_embeds,
                            text_ids,
                        ) = compute_text_embeddings(prompts, text_encoders, tokenizers)

                    # Convert images to latent space
                    if lora_config.cache_latents:
                        model_input = latents_cache[step].sample()
                    else:
                        pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                        model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (
                        model_input - vae_config_shift_factor
                    ) * vae_config_scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)

                    vae_scale_factor = 2 ** len(vae.config.block_out_channels)
                    # vae_scale_factor = 2 ** (len(vae.config.block_out_channels) -1 )

                    latent_image_ids = BriaPipeline._prepare_latent_image_ids(
                        model_input.shape[0],
                        model_input.shape[2],
                        model_input.shape[3],
                        accelerator.device,
                        weight_dtype,
                    )
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]

                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=lora_config.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=lora_config.logit_mean,
                        logit_std=lora_config.logit_std,
                        mode_scale=lora_config.mode_scale,
                    )
                    indices = (
                        u * noise_scheduler_copy.config.num_train_timesteps
                    ).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(
                        device=model_input.device
                    )

                    # Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(
                        timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                    )
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    packed_noisy_model_input = BriaPipeline._pack_latents(
                        noisy_model_input,
                        batch_size=model_input.shape[0],
                        num_channels_latents=model_input.shape[1],
                        height=model_input.shape[2],
                        width=model_input.shape[3],
                    )

                    # handle guidance
                    if transformer.config.guidance_embeds:
                        guidance = torch.tensor(
                            [lora_config.guidance_scale], device=accelerator.device
                        )
                        guidance = guidance.expand(model_input.shape[0])
                    else:
                        guidance = None

                    model_pred = transformer(
                        hidden_states=packed_noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]

                    model_pred = BriaPipeline._unpack_latents(
                        model_pred,
                        height=int(model_input.shape[2] * vae_scale_factor / 2),
                        width=int(model_input.shape[3] * vae_scale_factor / 2),
                        vae_scale_factor=vae_scale_factor,
                    )
                    # these weighting schemes use a uniform timestep sampling
                    # and instead post-weight the loss
                    weighting = compute_loss_weighting_for_sd3(
                        weighting_scheme=lora_config.weighting_scheme, sigmas=sigmas
                    )

                    # flow matching loss
                    # target = noise - model_input ##### TODO
                    target = model_input
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                    if lora_config.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        prior_loss = F.mse_loss(
                            model_pred_prior.to(torch.float32),
                            target_prior.to(torch.float32),
                            reduction="mean",
                        )

                    # Compute regular loss.
                    loss = torch.mean(
                        (
                            weighting.float()
                            * (model_pred.float() - target.float()) ** 2
                        ).reshape(target.shape[0], -1),
                        1,
                    )
                    loss = loss.mean()

                    if lora_config.with_prior_preservation:
                        # Add the prior loss to the instance loss.
                        loss = loss + lora_config.prior_loss_weight * prior_loss

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = transformer.parameters()
                        accelerator.clip_grad_norm_(
                            params_to_clip, lora_config.max_grad_norm
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % trainer_config.checkpoint_every_n_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if trainer_config.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(trainer_config.output_dir)
                                checkpoints = [
                                    d for d in checkpoints if d.startswith("checkpoint")
                                ]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if (
                                    len(checkpoints)
                                    >= trainer_config.checkpoints_total_limit
                                ):
                                    num_to_remove = (
                                        len(checkpoints)
                                        - trainer_config.checkpoints_total_limit
                                        + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            trainer_config.output_dir,
                                            removing_checkpoint,
                                        )
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(
                                trainer_config.output_dir, f"checkpoint-{global_step}"
                            )
                            accelerator.save_state(save_path)

                            logger.info(f"Saved state to {save_path}")

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= lora_config.max_train_steps:
                    break

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            transformer = unwrap_model(transformer)
            if trainer_config.upcast_before_saving:
                transformer.to(torch.float32)
            else:
                transformer = transformer.to(weight_dtype)
            transformer_lora_layers = get_peft_model_state_dict(transformer)

            text_encoder_lora_layers = None

            BriaPipeline.save_lora_weights(
                save_directory=trainer_config.output_dir,
                transformer_lora_layers=transformer_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
            )

        accelerator.end_training()
