from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    """
    Base configuration for distributed training strategies.
    """

    strategy_name: str = Field(
        ..., description="The name of the training strategy (e.g., 'FSDP', 'DDP')."
    )
    devices: int = Field(1, description="Number of devices to use.")
    mixed_precision: bool = Field(False, description="Enable mixed-precision training.")
    gradient_clipping: Optional[float] = Field(
        None, description="Maximum norm for gradient clipping."
    )
    additional_params: Optional[Dict[str, Any]] = Field(
        None, description="Additional parameters specific to the strategy."
    )
    compile: bool = Field(False, description="Compile the model for faster execution.")


class FSDPConfig(StrategyConfig):
    """
    Fully Sharded Data Parallel (FSDP) Configuration.

    Refer to: https://pytorch.org/docs/stable/fsdp.html
    """

    sharding_strategy: Optional[str] = Field(
        None,
        description="Sharding strategy for FSDP (e.g., 'FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD').",
    )
    offload_params: bool = Field(
        False, description="Enable parameter offloading to CPU."
    )
    cpu_offload: bool = Field(True, description="Offload gradients to CPU.")
    auto_wrap_policy: Optional[str] = Field(
        None, description="Policy for auto-wrapping layers for FSDP."
    )
    sync_module_states: bool = Field(
        True,
        description="Synchronize module states across workers during initialization.",
    )
    backward_prefetch: Optional[str] = Field(
        None,
        description="Backward prefetch mode (e.g., 'BACKWARD_PRE', 'BACKWARD_POST').",
    )
    activation_checkpointing: bool = Field(
        False,
        description="Enable activation checkpointing to save memory during training.",
    )


class DDPConfig(StrategyConfig):
    """
    Distributed Data Parallel (DDP) Configuration.

    Refer to: https://pytorch.org/docs/stable/ddp.html
    """

    find_unused_parameters: bool = Field(
        False, description="Find and handle unused parameters in the model."
    )
    bucket_cap_mb: Optional[int] = Field(
        None, description="Bucket size for DDP communication (in megabytes)."
    )
    gradient_as_bucket_view: bool = Field(
        True, description="Use gradient bucket views to reduce memory usage."
    )
    static_graph: bool = Field(
        False, description="Enable static graph optimization for DDP."
    )


class DPConfig(StrategyConfig):
    """
    Data Parallel (DP) Configuration.
    """

    batch_split: Optional[int] = Field(
        None, description="Split batch size across devices for DP."
    )


class SingleDeviceConfig(StrategyConfig):
    """
    Single-Device Training Configuration.
    """

    device_id: Optional[int] = Field(
        None, description="Device ID to use for single-device training."
    )
