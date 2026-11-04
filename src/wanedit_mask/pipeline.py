import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def FlowEditWan(
    pipe,
    scheduler,
    x_src_video,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 25,
    tar_guidance_scale: float = 7.5,
    n_max: int = 24,
    mask=None,
):
    """
    Flow Edit implementation for Wan2.1 with optional mask.

    Args:
        pipe: Wan2.1 pipeline
        scheduler: The scheduler to use
        x_src_video: Source video in latent space [B, C, F, H, W]
        src_prompt: Source prompt describing the input video
        tar_prompt: Target prompt describing desired output
        negative_prompt: Negative prompt
        T_steps: Total diffusion steps (default: 25 for Wan)
        tar_guidance_scale: Guidance scale for target prompt (default: 7.5 for Wan)
        n_max: Maximum step to apply editing (default: 24 for Wan)
        mask: Optional binary mask [1, 1, F, H, W] in latent space

    Returns:
        Edited video latents
    """

    device = x_src_video.device

    # 1. Prepare timesteps
    scheduler.set_timesteps(T_steps, device=device)
    timesteps = scheduler.timesteps

    # 2. Encode prompts
    # Source prompt
    src_prompt_embeds = pipe.encode_prompt(
        prompt=src_prompt,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
    )[0]

    # Target prompt (with CFG)
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

        scheduler._init_step_index(t)

        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1]
        else:
            t_im1 = torch.zeros_like(t).to(t.device)

        # Normalize timesteps to [0, 1] range
        t_i = t / 1000.0
        t_im1 = t_im1 / 1000.0

        # Forward process with single noise sample
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

        # Predict V for target
        with torch.no_grad():
            # Target prediction
            Vt_tar = pipe.transformer(
                hidden_states=zt_tar,
                timestep=timestep_expanded,
                encoder_hidden_states=tar_prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]

            # Apply guidance for target
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

        # Apply mask if provided
        if mask is not None:
            V_delta = V_delta * mask

        # Update ODE trajectory
        zt_edit = zt_edit.to(torch.float32)
        zt_edit = zt_edit + (t_im1 - t_i) * V_delta
        zt_edit = zt_edit.to(V_delta.dtype)

    # Return edited video latents
    return zt_edit
