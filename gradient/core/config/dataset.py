from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """
    Configuration of PyTorch Dataset.

    For details on the function/meanings of the arguments, refer to:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    dataset_name: Optional[str] = Field(
        default=None,
        description=(
            "The name of the Dataset (from the HuggingFace hub) to train on. It can also be a path pointing to a local "
            "copy of a dataset in your filesystem, or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    dataset_config_name: Optional[str] = Field(
        default=None,
        description="The config of the Dataset, leave as None if there's only one config.",
    )
    data_channels: str = Field(
        default="train_1",
        description="A folder containing the training data dirs separated by commas.",
    )
    image_column: str = Field(
        default="image",
        description="The column of the dataset containing an image.",
    )
    caption_column: str = Field(
        default="text",
        description="The column of the dataset containing a caption or a list of captions.",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Path to a directory where the dataset will be cached.",
    )
    train_batch_size: int = Field(
        default=2,
        description="Number of samples per batch to load.",
    )
    random_flip: bool = Field(
        default=False,
        description="Randomly flip the images horizontally.",
    )
    center_crop: bool = Field(
        default=False,
        description="Center crop the images.",
    )
    resolution: int = Field(
        default=256,
        description="Resolution of the images.",
    )
