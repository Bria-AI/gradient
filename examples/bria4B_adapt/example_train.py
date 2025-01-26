import datetime
from core.config.bria4B_adapt import HyperParemeter as model_config
from core.config.dataloader import DataLoaderConfig
from core.config.loggers import WandB
from core.config.dataset import DatasetConfig
from core.config.startegy import FSDPConfig
from models.diffusion.bria4B_adapt.train import Bria4BAdapt as Model
from core.config.trainer import TrainerConfig

if __name__ == "__main__":
    print("Running example_train.py")
    bria_conf = model_config(
        train_batch_size=1,
        precompute=False,
        dense_caption_ratio=0.8,  # 1
        gradient_accumulation_steps=1,
        flow_matching_latent_loss=1,
        shift=3.0,
        use_flow_matching=1,
        use_dynamic_shift=0,
        train_with_ratios=0,
        resolution=1024,
        resize=True,
        center_crop=True,
        weighting_scheme="uniform",
        force_download=False,
        use_8bit_adam=True,
    )


    model = Model(bria_conf)

    dataset = DatasetConfig(
        dataset_name="timm/imagenet-1k-wds",
        train_batch_size=1,
        caption_column="text",
        image_column="image",
    )

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    logger = WandB(
        wandb_mode="online",
        wandb_project="pulse-project",
        wandb_group="sagemaker",
        wandb_run_name=f"sagemaker-test-run-{time}",
    )

    # Trainer
    trainer = TrainerConfig(
        max_train_steps=1001,
        # resume_from_checkpoint="/home/ubuntu/gradient/checkpoints/BRIA-4B-Adapt",
        # base_model_dir="/gradient/checkpoints/transformer",
        huggingface_path="briaai/BRIA-4B-Adapt",
        checkpoint_local_path="./checkpoints",
        checkpointing_steps=250,
    )

    # Startegey
    strategy = FSDPConfig(
        strategy_name="FSDP",
        cpu_offload=True,
    )

    # Train
    model.train(
        dataset_config=dataset,
        trainer_config=trainer,
        logger_config=logger,
        startegy_config=strategy,
        dataloader_config=DataLoaderConfig(num_workers=1, batch_size=1),
    )
