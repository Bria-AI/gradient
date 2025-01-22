from typing import Any, Optional

from pydantic import BaseModel, Field


class DataLoaderConfig(BaseModel):
    """
    Configuration of PyTorch DataLoader.

    For details on the function/meanings of the arguments, refer to:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    batch_size: int = Field(..., description="Number of samples per batch to load.")
    shuffle: bool = Field(
        False, description="Set to True to have the data reshuffled at every epoch."
    )
    sampler: Optional[Any] = Field(
        None, description="Defines the strategy to draw samples from the dataset."
    )
    batch_sampler: Optional[Any] = Field(
        None, description="Sampler to draw batches directly."
    )
    num_workers: int = Field(
        0, description="Number of subprocesses to use for data loading."
    )
    collate_fn: Optional[Any] = Field(
        None, description="Function to merge a list of samples to form a mini-batch."
    )
    pin_memory: bool = Field(
        False,
        description="If True, the data loader will copy Tensors into CUDA pinned memory.",
    )
    drop_last: bool = Field(
        False,
        description="Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.",
    )
    timeout: int = Field(
        0, description="Timeout value in seconds for collecting a batch."
    )
    worker_init_fn: Optional[Any] = Field(
        None,
        description="If not None, this function will be called on each worker subprocess.",
    )
    multiprocessing_context: Optional[Any] = Field(
        None, description="Context for multiprocessing."
    )
