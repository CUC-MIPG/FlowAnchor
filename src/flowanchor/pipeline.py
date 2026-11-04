import math

import torch
import numpy as np
from tqdm import tqdm
from .utils import get_diff_mask
from . import ptp_utils


@torch.no_grad()
def FlowEditWan(
    pipe,
    scheduler,
    x_src_video,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 25,
    tar_guidance_scale: float = 1.0,
    n_max: int = 23,
    omega: float = 1.0,
    ca_mask: torch.Tensor = None,
    target_word: str = None,
    row_strength: float = 0.3,
    col_strength: float = 0.3,
    edit_mask: torch.Tensor = None,
):
    """
    Flow Edit implementation for Wan2.1 with dynamic mask weighting and CA modulation.

    Args:
        pipe: Wan2.1 pipeline
        scheduler: The scheduler to use
        x_src_video: Source video in latent space [B, C, F, H, W]
        src_prompt: Source prompt describing the input video
        tar_prompt: Target prompt describing desired output
        negative_prompt: Negative prompt
        T_steps: Total diffusion steps (default: 25 for Wan)
        tar_guidance_scale: Guidance scale for target prompt (default: 1.0 for Wan)
        n_max: Number of active editing steps (default: 23)
        omega: Base AMM strength, scaled by log(F) / log(21) for latent frame count F
        ca_mask: CA resolution mask [1, 1, F_ca, H_ca, W_ca] (optional)
        target_word: Target word/phrase for CA modulation (optional)
        row_strength: Row modulation strength (default: 0.3)
        col_strength: Column modulation strength (default: 0.3)
        edit_mask: Spatial mask [B, 1, F, H, W] or [1, 1, F, H, W] for constraining edits (optional)
                   Values should be in [0, 1] where 1 means fully editable

    Returns:
        Edited video latents
    """

    device = x_src_video.device
    # Wan2.1 encodes the 81-frame reference video into 21 latent frames.
    num_latent_frames = x_src_video.shape[2]
    omega_f = omega * (math.log(num_latent_frames) / math.log(21))

    # Prepare edit mask if provided
    if edit_mask is not None:
        # Ensure mask has correct shape [B, 1, F, H, W] or [1, 1, F, H, W]
        if edit_mask.dim() == 4:  # [B, F, H, W]
            edit_mask = edit_mask.unsqueeze(1)
        elif edit_mask.dim() == 3:  # [F, H, W]
            edit_mask = edit_mask.unsqueeze(0).unsqueeze(0)

        # Move to correct device and ensure values are in [0, 1]
        edit_mask = edit_mask.to(device=device, dtype=x_src_video.dtype)
        edit_mask = torch.clamp(edit_mask, 0.0, 1.0)
        print(f"Edit mask enabled with shape: {edit_mask.shape}")

    # Setup CA modulation config (only if needed)
    modulation_config = None

    if ca_mask is not None and target_word is not None:
        target_word_ids = ptp_utils.get_word_ids(
            tar_prompt, target_word, pipe.tokenizer
        )

        if len(target_word_ids) > 0:
            modulation_config = {
                "ca_mask": ca_mask,
                "target_word_ids": target_word_ids,
                "modulate_layers": list(range(30)),
                "modulate_steps": list(range(2, 10)),  # steps to apply modulation
                "current_step": -1,  # will be updated in loop
                "row_strength": row_strength,
                "col_strength": col_strength,
            }
            print(
                f"CA modulation enabled: target_word='{target_word}', token_ids={target_word_ids}"
            )
            print(
                f"Modulation will be applied at steps: {modulation_config['modulate_steps']}"
            )
        else:
            print(f"Warning: target_word '{target_word}' not found in prompt")

    # 1. Prepare timesteps
    scheduler.set_timesteps(T_steps, device=device)
    timesteps = scheduler.timesteps

    # 2. Encode prompts
    src_prompt_embeds = pipe.encode_prompt(
        prompt=src_prompt,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
    )[0]

    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
    )

    # 3. Initialize our ODE with source video
    zt_edit = x_src_video.clone()

    # 4. Denoising loop
    for i, t in tqdm(enumerate(timesteps)):

        if T_steps - i > n_max:
            continue

        # Update modulation config with current step
        if modulation_config is not None:
            modulation_config["current_step"] = i

        scheduler._init_step_index(t)

        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1]
        else:
            t_im1 = torch.zeros_like(t).to(t.device)

        t_i = t / 1000.0
        t_im1 = t_im1 / 1000.0

        fwd_noise = torch.randn_like(x_src_video).to(x_src_video.device)
        zt_src = (1 - t_i) * x_src_video + t_i * fwd_noise

        zt_tar = zt_edit + zt_src - x_src_video

        # Predict V for source
        with torch.no_grad():
            timestep_expanded = t.expand(zt_src.shape[0])
            Vt_src = pipe.transformer(
                hidden_states=zt_src,
                timestep=timestep_expanded,
                encoder_hidden_states=src_prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]

        # Predict V for target (with CA modulation if needed)
        with torch.no_grad():
            # Only register hooks if we need to modulate at this step
            should_modulate = (
                modulation_config is not None
                and i in modulation_config["modulate_steps"]
            )

            if should_modulate:
                # Register hook with modulation config
                ptp_utils.register_attention_control(
                    pipe.transformer, modulation_config
                )

            # Target conditional prediction
            Vt_tar = pipe.transformer(
                hidden_states=zt_tar,
                timestep=timestep_expanded,
                encoder_hidden_states=tar_prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]

            # Restore original processors if we modulated
            if should_modulate:
                for block in pipe.transformer.blocks:
                    block.attn2.processor = block.attn2.processor.original_processor

            tar_noise_uncond = pipe.transformer(
                hidden_states=zt_tar,
                timestep=timestep_expanded,
                encoder_hidden_states=tar_negative_prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]

            Vt_tar = tar_noise_uncond + tar_guidance_scale * (Vt_tar - tar_noise_uncond)

        # Calculate flow field difference
        V_delta = Vt_tar - Vt_src

        # Generate dynamic mask and apply weighting
        diff_mask = get_diff_mask(V_delta)
        V_delta = (1.0 + omega_f * diff_mask) * V_delta

        # # Apply edit mask: only update within masked regions
        # if edit_mask is not None:
        #     V_delta = edit_mask * V_delta

        # Update ODE trajectory
        zt_edit = zt_edit.to(torch.float32)
        zt_edit = zt_edit + (t_im1 - t_i) * V_delta
        zt_edit = zt_edit.to(V_delta.dtype)

    return zt_edit
