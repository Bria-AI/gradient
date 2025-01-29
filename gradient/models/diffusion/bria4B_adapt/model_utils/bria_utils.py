from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.utils import logging
from transformers import T5EncoderModel, T5TokenizerFast

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_t5_prompt_embeds(
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 128,
    device: Optional[torch.device] = None,
):
    device = device or text_encoder.device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        # padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(
        prompt, padding="longest", return_tensors="pt"
    ).input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, max_sequence_length - 1 : -1]
        )
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    # Concat zeros to max_sequence
    b, seq_len, dim = prompt_embeds.shape
    if seq_len < max_sequence_length:
        padding = torch.zeros(
            (b, max_sequence_length - seq_len, dim),
            dtype=prompt_embeds.dtype,
            device=prompt_embeds.device,
        )
        prompt_embeds = torch.concat([prompt_embeds, padding], dim=1)

    prompt_embeds = prompt_embeds.to(device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# in order the get the same sigmas as in training and sample from them
def get_original_sigmas(num_train_timesteps=1000, num_inference_steps=1000):
    timesteps = np.linspace(
        1, num_train_timesteps, num_train_timesteps, dtype=np.float32
    )[::-1].copy()
    sigmas = timesteps / num_train_timesteps

    inds = [
        int(ind) for ind in np.linspace(0, num_train_timesteps - 1, num_inference_steps)
    ]
    new_sigmas = sigmas[inds]
    return new_sigmas


def is_ng_none(negative_prompt):
    return (
        negative_prompt is None
        or negative_prompt == ""
        or (isinstance(negative_prompt, list) and negative_prompt[0] is None)
        or (type(negative_prompt) == list and negative_prompt[0] == "")
    )
