import torch
import torch.nn.functional as nnf
import cv2
import numpy as np
import os


def load_video_to_frames(video_path):
    """
    Load a video file into frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Ensure frame is uint8
        frame = frame.astype(np.uint8)
        frames.append(frame)

    cap.release()
    if not frames:
        raise ValueError(f"Video contains no readable frames: {video_path}")
    return frames


def encode_video_to_latents(pipe, video_tensor):
    """
    Encode video frames to latent space using Wan2.1's VAE
    """
    device = pipe.device
    dtype = pipe.vae.dtype
    video_tensor = video_tensor.to(device).to(dtype)

    # Process in chunks according to vae_scale_factor_temporal
    vae_scale_factor_temporal = pipe.vae_scale_factor_temporal

    # Encode frames to latents
    with torch.no_grad():
        video_latents = pipe.vae.encode(video_tensor).latent_dist.sample()

        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(device, dtype)
        video_latents = (video_latents - latents_mean) * latents_std

    return video_latents


def load_masks(mask_dir, num_frames):
    """
    Load binary masks for each frame

    Args:
        mask_dir: Directory containing binary mask images (00000.png, 00001.png, ...)
        num_frames: Number of frames to load

    Returns:
        mask_tensor: [1, 1, F, H, W] tensor with values in [0, 1]
    """
    masks = []
    for i in range(num_frames):
        mask_path = os.path.join(mask_dir, f"{i:05d}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if mask is None:
            raise ValueError(f"Missing or unreadable mask: {mask_path}")
        if masks and mask.shape != masks[0].shape:
            raise ValueError(f"Inconsistent mask dimensions: {mask_path}")
        masks.append(mask)

    mask_array = np.stack(masks, axis=0)  # [F, H, W]
    mask_tensor = torch.from_numpy(mask_array).float() / 255.0  # Normalize to [0, 1]
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, F, H, W]
    return mask_tensor


def resize_mask_to_latent(mask_tensor, latent_shape):
    """
    Resize mask to latent space dimensions and apply dilation

    Args:
        mask_tensor: [1, 1, F, H, W] mask tensor
        latent_shape: [B, C, F, H_latent, W_latent] shape of latent

    Returns:
        latent_mask: [1, 1, F, H_latent, W_latent] dilated binary mask in latent space
    """
    _, _, f_latent, h_latent, w_latent = latent_shape

    # Resize to latent dimensions using trilinear interpolation
    mask_resized = nnf.interpolate(
        mask_tensor,
        size=(f_latent, h_latent, w_latent),
        mode="trilinear",
        align_corners=False,
    )  # [1, 1, F, H_latent, W_latent]

    # Apply 3x3 max pooling dilation frame by frame
    mask_dilated = torch.stack(
        [
            nnf.max_pool2d(
                mask_resized[:, :, f, :, :], kernel_size=3, stride=1, padding=1
            )
            for f in range(f_latent)
        ],
        dim=2,
    )  # [1, 1, F, H_latent, W_latent]

    # Binarize
    latent_mask = (mask_dilated > 0.5).float()

    return latent_mask


def get_diff_mask(V_delta):
    """
    Generate dynamic mask from velocity difference

    Args:
        V_delta: [B, C, F, H, W] velocity difference tensor

    Returns:
        diff_mask: [B, 1, F, H, W] normalized mask in [0, 1]
    """
    B = V_delta.shape[0]

    # Average over channel dimension
    mask = V_delta.mean(dim=1, keepdim=True)  # [B, 1, F, H, W]

    # Min-Max normalization per sample
    mask_min = mask.reshape(B, -1).min(dim=1).values.reshape(B, 1, 1, 1, 1)
    mask_max = mask.reshape(B, -1).max(dim=1).values.reshape(B, 1, 1, 1, 1)
    diff_mask = (mask - mask_min) / (mask_max - mask_min + 1e-7)

    return diff_mask


def show_diff_mask(diff_mask, save_path="./diff_mask/mask.png"):
    """
    Visualize diff_mask per frame as grayscale images

    Args:
        diff_mask: [B, 1, F, H, W] normalized mask tensor in [0, 1]
        save_path: base path to save figures (e.g. "./diff_mask/mask.png")
                   will create mask_frame_0.png, mask_frame_1.png, etc.
    """
    import os
    from PIL import Image

    # Remove batch and channel dimensions: [B, 1, F, H, W] -> [F, H, W]
    diff_mask = diff_mask.squeeze(0).squeeze(0).cpu().numpy()  # [F, H, W]

    F, H, W = diff_mask.shape

    # Prepare save directory
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # Save each frame as grayscale image
    for frame_idx in range(F):
        mask_frame = diff_mask[frame_idx]  # [H, W], values in [0, 1]

        mask_uint8 = (mask_frame * 255).astype(np.uint8)

        frame_save_path = save_path.replace(".png", f"_frame_{frame_idx}.png")
        Image.fromarray(mask_uint8, mode="L").save(frame_save_path)

    print(f"Saved {F} diff mask frames to {save_dir}/")
