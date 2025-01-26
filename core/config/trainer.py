from typing import Optional
from pathlib import Path
import os
from huggingface_hub import HfApi, HfFolder, Repository, repo_exists
from pydantic import BaseModel, Field


class TrainerConfig(BaseModel):
    """
    Configuration for the Trainer.
    """

    max_train_steps: int = Field(
        default=250000,
        description="Total number of training steps to perform.",
    )
    resume_from_checkpoint: str = Field(
        default="no",
        description="Whether training should be resumed from a previous checkpoint.",
    )
    log_dir: str = Field("logs/", description="Directory to save logs.")
    checkpoint_every_n_steps: int = Field(
        1000, description="Save a checkpoint every n steps."
    )
    checkpoint_local_path: str = Field(
        default="/tmp/checkpoints",
        description="The local path to save checkpoints to. Then SageMaker will upload from there to S3.",
    )
    base_model_dir: str = Field(
        default="/tmp/models",
        description="The local of the model.",
    )
    checkpointing_steps: int = Field(
        default=5000,
        description="Save a checkpoint of the training state every X updates.",
    )
    num_train_epochs: int = Field(
        default=100,
        description="Number of training epochs.",
    )
    output_dir: str = Field(
        default="/tmp/output",
        description="A path to a directory for storing data.",
    )
    checkpoints_total_limit: Optional[int] = Field(
        default=None,
        description="Maximum number of checkpoints to store.",
    )
    upcast_before_saving: Optional[bool] = Field(
        default=False,
        description="Upcast the model before saving.",
    )
    huggingface_path: Optional[str] = Field(
        default=None,
        description="Path to a directory containing a Hugging Face model.",
    )

    def _download_checkpoint(self, resume_from_checkpoint: str) -> None:
        """
        Download a checkpoint from Hugging Face Hub.
        """
        if self.resume_from_checkpoint == "no" or self.resume_from_checkpoint is None:
            self.checkpoint_dir = "no"

        if not repo_exists(resume_from_checkpoint) and os.access(
            os.path.dirname(resume_from_checkpoint), os.W_OK
        ):
            self.checkpoint_dir = resume_from_checkpoint
            return
        if repo_exists(resume_from_checkpoint):
            self.from_huggingface_hub = True
            return
