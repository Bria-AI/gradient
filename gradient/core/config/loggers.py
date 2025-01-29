from pydantic import BaseModel, Field


class Logger(BaseModel):
    logging_dir: str = Field(
        default="logs",
        description="log directory.",
    )
    report_to: str = Field(default="wandb")


class WandB(Logger):
    wandb_mode: str = Field(
        default="online",
        description="Enable or disable WandB.",
    )
    wandb_project: str = Field(
        default="default",
        description="WandB project name.",
    )
    wandb_entity: str = Field(
        default=None,
        description="WandB entity name.",
    )
    wandb_run_name: str = Field(
        default=None,
        description="WandB run name.",
    )
    wandb_group: str = Field(
        default=None,
        description="WandB group name.",
    )
    wandb_tags: list = Field(
        default=[],
        description="WandB tags.",
    )
    save_images_to_wandb: bool = Field(
        default=False,
        description="Save images and latents to WandB according to save_images_every.",
    )
    report_to: str = Field(
        default="wandb",
        description="The integration to report results and logs.",
    )
    save_images_every: int = Field(
        default=1,
        description="How many steps to save images for WandB.",
    )
