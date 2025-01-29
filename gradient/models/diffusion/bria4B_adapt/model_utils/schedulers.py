# Copyright 2024 Stanford University Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion


from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.configuration_utils import register_to_config


def shift_noise_schedule(alphas_cumprod, shift):
    alphas_cumprod = alphas_cumprod.clone()
    if shift > 1:
        snr = alphas_cumprod / (1 - alphas_cumprod)
        log_snr = torch.log(snr)
        shifted_log_snr = log_snr + torch.log(1 / torch.tensor(shift))
        alphas_cumprod = torch.sigmoid(shifted_log_snr)

    return alphas_cumprod


class BriaDDIMScheduler(DDIMScheduler):
    """
    BriaDDIMScheduler extends DDIMScheduler with noise shifting
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
        shift: float = 1.0,
    ):

        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            clip_sample,
            set_alpha_to_one,
            steps_offset,
            prediction_type,
            thresholding,
            dynamic_thresholding_ratio,
            clip_sample_range,
            sample_max_value,
            timestep_spacing,
            rescale_betas_zero_snr,
        )
        # Theoratically for for high res n relative to low res m we need to shift the logsnr by log((m/n)^2) to maintaing low res statistics
        # In practice we shift it by a value of log(2+)
        if shift > 1:
            self.alphas_cumprod = shift_noise_schedule(self.alphas_cumprod, shift)
            # These are invalidated
            self.alphas = None
            self.betas = None


class BriaEulerAncestralDiscreteScheduler(EulerAncestralDiscreteScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        shift: float = 1.0,
    ):
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            prediction_type,
            timestep_spacing,
            steps_offset,
            rescale_betas_zero_snr,
        )

        # Theoratically for for high res n relative to low res m we need to shift the logsnr bt log((m/n)^2) to maintaing low res statistics
        # In practice we shift it by a value of 2+
        if shift > 1:
            self.alphas_cumprod = shift_noise_schedule(self.alphas_cumprod, shift)
            # These are invalidated
            self.alphas = None
            self.betas = None

        if rescale_betas_zero_snr:
            # Close to 0 without being 0 so first sigma is not inf
            # FP16 smallest positive subnormal works well here
            self.alphas_cumprod[-1] = 2**-24

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
