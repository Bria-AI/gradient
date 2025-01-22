import os
import sys
import datetime
from core.config.bria4B_adapt import HyperParemeter as model_config
from core.config.dataloader import DataLoaderConfig
from core.config.loggers import WandB
from core.config.dataset import DatasetConfig
from core.config.startegy import FSDPConfig
from models.diffusion.bria4B_adapt.train import Bria4BAdapt as Model
from core.config.trainer import TrainerConfig

# SageMaker environment variables
checkpoint_local_path = os.environ.get("SM_MODEL_DIR", "/opt/ml/checkpoints")
training_data_path = os.environ.get("SM_CHANNEL_TRAINING", "./data/training")
output_data_path = os.environ.get("SM_OUTPUT_DATA_DIR", "./output")
region_name = os.environ.get("AWS_REGION", "us-east-1")

bria_conf = model_config(
    train_batch_size=1,
    precompute=False,
    dense_caption_ratio=0.8,
    gradient_accumulation_steps=1,
    flow_matching_latent_loss=1,
    shift=3.0,
    use_flow_matching=1,
    use_dynamic_shift=0,
    train_with_ratios=0,
    resolution=256,
    resize=True,
    center_crop=True,
    weighting_scheme="uniform",
    force_download=False,
)

model = Model(bria_conf)

dataset = DatasetConfig(
    dataset_name="timm/imagenet-1k-wds",
    train_batch_size=1,
    caption_column="text",
    image_column="image",
    data_dir=training_data_path,  # Load data from SageMaker-provided path
)

dataloader = DataLoaderConfig(num_workers=1, batch_size=1)

time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

logger = WandB(
    wandb_mode="offline",
    wandb_project="EA-tests",
    wandb_group="test-1",
    wandb_run_name=f"test-{time}",
)

trainer = TrainerConfig(
    max_train_steps=20,
    base_model_dir="checkpoints/BRIA-4B-Adapt",
    checkpoint_local_path=checkpoint_local_path,
    checkpointing_steps=10,
)

os.environ["HF_API_TOKEN"] = "'HUGGINGFACE_TOKEN'"

strategy = FSDPConfig(
    strategy_name="FSDP",
    cpu_offload=True,
)

# Train the model
model.train(
    dataset_config=dataset,
    trainer_config=trainer,
    logger_config=logger,
    startegy_config=strategy,
    dataloader_config=dataloader,
)
