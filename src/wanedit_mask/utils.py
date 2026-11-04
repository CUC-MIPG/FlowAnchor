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


def visualize_mask(latent_mask, save_dir="./latent_mask", filename="mask_frame0.png"):
    """
    Visualize the first frame of latent space mask

    Args:
        latent_mask: [1, 1, F, H_latent, W_latent] mask tensor in latent space
        save_dir: Directory to save visualization
        filename: Filename for saved image
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract first frame: [H_latent, W_latent]
    first_frame = latent_mask[0, 0, 0].cpu().numpy()
    mask_img = (first_frame * 255).astype(np.uint8)

    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, mask_img)
    print(f"Saved latent mask to: {save_path}")
