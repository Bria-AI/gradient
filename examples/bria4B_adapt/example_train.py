import datetime
import sys


from gradient.core.config.bria4B_adapt import HyperParemeter as model_config
from gradient.core.config.dataloader import DataLoaderConfig
from gradient.core.config.loggers import WandB
from gradient.core.config.dataset import DatasetConfig
from gradient.core.config.startegy import FSDPConfig
from gradient.models.diffusion.bria4B_adapt.train import Bria4BAdapt as Model
from gradient.core.config.trainer import TrainerConfig

if __name__ == "__main__":
    print("Running example_train.py")
    bria_conf = model_config(
        train_batch_size=3,
        precompute=False,
        dense_caption_ratio=1.0,  # 1
        gradient_accumulation_steps=1,
        flow_matching_latent_loss=0,
        shift=4.0,
        mixed_precision="bf16",
        # use_flow_matching=1, #?
        # use_dynamic_shift=0,
        # train_with_ratios=0,
        lr_warmup_steps=100,  # ?
        resolution=1024,
        resize=True,
        center_crop=True,
        weighting_scheme="uniform",
        force_download=False,
        max_grad_norm=1.0,
        adam_weight_decay=1e-04,
        adam_epsilon=1e-08,
        learning_rate=5e-05,
        random_latents=0,
        train_with_ratios=0,
        reinit_scheduler=0,
        reinit_optimizer=0,
        s3_bucket_name="eiga-datasets",
        s3_prefix="fox_for_bria_adapt_test_square",
        pretrained_vae_model_name_or_path="briaai/BRIA-4B-Adapt",
        pretrained_text_encoder_name_or_path="briaai/BRIA-4B-Adapt"
    )

    model = Model(bria_conf)

    dataset = DatasetConfig(
        local_path="/home/ubuntu/datasets/fox",
        caption_column="answer",
        image_column="image",
        train_batch_size=3,

        
    )

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    logger = WandB(
        wandb_mode="offline",
        wandb_project="test",
        wandb_group="test-1",
        wandb_run_name=f"test-{time}",
    )

    # Trainer
    trainer = TrainerConfig(
        max_train_steps=1000,
        # resume_from_checkpoint="/home/ubuntu/gradient/checkpoints/BRIA-4B-Adapt",
        # base_model_dir="/home/ubuntu/BRIA-4B-Adapt/transformer",
        checkpoint_local_path="/home/ubuntu/gradient/checkpoints",
        checkpointing_steps=500,
        huggingface_path="briaai/BRIA-4B-Adapt",
    )

    # Startegey
    strategy = FSDPConfig(
        strategy_name="FSDP",
    )

    # Train
    model.train(
        dataset_config=dataset,
        trainer_config=trainer,
        logger_config=logger,
        startegy_config=strategy,
        dataloader_config=DataLoaderConfig(num_workers=1, batch_size=3),
    )
