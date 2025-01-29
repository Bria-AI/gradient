from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field


class OptimizerParams(BaseModel):
    """
    Base Optimizer params with no values. Users can explicitly override via
    command line arguments.
    """

    lr: Optional[float] = Field(None, description="Learning rate for the optimizer.")


class SGDParams(OptimizerParams):
    """
    Default configuration for SGD optimizer.

    Refer to: https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
    """

    momentum: float = Field(0.0, description="Momentum factor.")
    dampening: float = Field(0.0, description="Dampening for momentum.")
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")
    nesterov: bool = Field(False, description="Enables Nesterov momentum.")


class AdamParams(OptimizerParams):
    """
    Default configuration for Adam optimizer.

    Refer to: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    """

    eps: float = Field(
        1e-08, description="Term added to the denominator for numerical stability."
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")
    amsgrad: bool = Field(
        False, description="Whether to use the AMSGrad variant of the Adam optimizer."
    )


class AdamWParams(OptimizerParams):
    """
    Default configuration for AdamW optimizer.

    Refer to: https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW
    """

    betas: Tuple[float, float] = Field(
        (0.9, 0.999), description="Coefficients used for computing running averages."
    )
    eps: float = Field(
        1e-08, description="Term added to the denominator for numerical stability."
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")
    amsgrad: bool = Field(
        False, description="Whether to use the AMSGrad variant of the AdamW optimizer."
    )


class AdadeltaParams(OptimizerParams):
    """
    Default configuration for Adadelta optimizer.

    Refer to: https://pytorch.org/docs/stable/optim.html#torch.optim.Adadelta
    """

    rho: float = Field(
        0.9,
        description="Coefficient used for computing a running average of squared gradients.",
    )
    eps: float = Field(
        1e-6, description="Term added to the denominator for numerical stability."
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")


class AdamaxParams(OptimizerParams):
    """
    Default configuration for Adamax optimizer.

    Refer to: https://pytorch.org/docs/stable/optim.html#torch.optim.Adamax
    """

    betas: Tuple[float, float] = Field(
        (0.9, 0.999), description="Coefficients used for computing running averages."
    )
    eps: float = Field(
        1e-8, description="Term added to the denominator for numerical stability."
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")


class AdagradParams(OptimizerParams):
    """
    Default configuration for Adagrad optimizer.

    Refer to: https://pytorch.org/docs/stable/optim.html#torch.optim.Adagrad
    """

    lr_decay: float = Field(0.0, description="Learning rate decay.")
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")
    initial_accumulator_value: float = Field(
        0.0, description="Initial accumulator value."
    )
    eps: float = Field(
        1e-10, description="Term added to the denominator for numerical stability."
    )


class RMSpropParams(OptimizerParams):
    """
    Default configuration for RMSprop optimizer.

    Refer to: https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop
    """

    alpha: float = Field(0.99, description="Smoothing constant for running average.")
    eps: float = Field(
        1e-8, description="Term added to the denominator for numerical stability."
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")
    momentum: float = Field(0.0, description="Momentum factor.")
    centered: bool = Field(False, description="If True, compute the centered RMSProp.")


class RpropParams(OptimizerParams):
    """
    Default configuration for Rprop optimizer.

    Refer to: https://pytorch.org/docs/stable/optim.html#torch.optim.Rprop
    """

    etas: Tuple[float, float] = Field(
        (0.5, 1.2), description="Step increase and decrease factors."
    )
    step_sizes: Tuple[float, float] = Field(
        (1e-6, 50), description="Minimum and maximum allowed step sizes."
    )


class NovogradParams(OptimizerParams):
    """
    Configuration for the Novograd optimizer.

    Refer to: https://arxiv.org/abs/1905.11286
    """

    betas: Tuple[float, float] = Field(
        (0.95, 0.98), description="Coefficients for running averages."
    )
    eps: float = Field(
        1e-8, description="Term added to the denominator for numerical stability."
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")
    grad_averaging: bool = Field(
        False, description="Whether to use gradient averaging."
    )
    amsgrad: bool = Field(
        False, description="Whether to use the AMSGrad variant of the optimizer."
    )
    luc: bool = Field(False, description="Enable Layer-wise Update Clipping (LUC).")
    luc_trust: float = Field(1e-3, description="Trust factor for LUC.")
    luc_eps: float = Field(1e-8, description="Epsilon value for LUC.")


class AdafactorParams(OptimizerParams):
    """
    Configuration for the Adafactor optimizer.

    Refer to: https://arxiv.org/abs/1804.04235
    """

    beta1: Optional[float] = Field(
        None, description="Coefficient for computing running averages of gradients."
    )
    eps: Tuple[float, float] = Field(
        (1e-30, 1e-3), description="Epsilon values for numerical stability."
    )
    clip_threshold: float = Field(1.0, description="Clipping threshold for gradients.")
    decay_rate: float = Field(
        0.8, description="Decay rate for second moment estimates."
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")
    scale_parameter: bool = Field(True, description="Enable parameter scaling.")
    relative_step: bool = Field(False, description="Enable relative step size.")
    warmup_init: bool = Field(
        False, description="Enable learning rate warmup initialization."
    )
