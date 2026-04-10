import argparse
import glob
import os
import re
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from examples.wanvideo.model_training.WanTrainingModule import \
    WanTrainingModule
try:
    from test_script.exr import (filter_frame_range, format_sequence_path,
                                 gather_exr_sequence, linear_to_srgb,
                                 read_exr_rgb, save_depth_exr)
except ModuleNotFoundError:
    from exr import (filter_frame_range, format_sequence_path,
                     gather_exr_sequence, linear_to_srgb, read_exr_rgb,
                     save_depth_exr)


# =============================
# Helper: Math & Alignment
# =============================
def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
    """Computes scale and shift for overlap alignment."""
    if mask is None:
        mask = np.ones_like(ref_frames)

    a_00 = np.sum(mask * curr_frames * curr_frames)
    a_01 = np.sum(mask * curr_frames)
    a_11 = np.sum(mask)
    b_0 = np.sum(mask * curr_frames * ref_frames)
    b_1 = np.sum(mask * ref_frames)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale, shift = 1.0, 0.0

    return scale, shift


def depth_to_single_channel(depth):
    if isinstance(depth, torch.Tensor):
        if depth.ndim != 5:
            raise ValueError(f"Expected depth output with 5 dims, got shape {tuple(depth.shape)}")
        if depth.shape[-1] == 1:
            return depth
        if depth.shape[-1] != 3:
            raise ValueError(
                f"Expected depth output with 1 or 3 channels, got shape {tuple(depth.shape)}"
            )
        return depth[..., :1]

    if depth.ndim != 5:
        raise ValueError(f"Expected depth output with 5 dims, got shape {depth.shape}")
    if depth.shape[-1] == 1:
        return depth
    if depth.shape[-1] != 3:
        raise ValueError(f"Expected depth output with 1 or 3 channels, got shape {depth.shape}")
    return depth[..., :1]


def _read_frame_worker(args):
    path, width, height = args
    frame_np = read_exr_rgb(path)
    frame_np = frame_np / (1.0 + frame_np)  # suppress overbrights
    frame_np = np.clip(linear_to_srgb(frame_np), 0.0, 1.0)

    if width is not None and height is not None:
        orig_height, orig_width = frame_np.shape[:2]
        if width * height < orig_width * orig_height:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        frame_np = cv2.resize(frame_np, (width, height), interpolation=interpolation)

    return frame_np.astype(np.float32)


def read_exr_sequence(paths, width=None, height=None, workers=None):
    if workers is None:
        workers = min(os.cpu_count() or 4, len(paths))

    work = [(path, width, height) for path in paths]

    if len(paths) <= 2:
        frames = [_read_frame_worker(args) for args in tqdm(work, desc="Reading EXRs")]
    else:
        with Pool(workers) as pool:
            frames = list(
                tqdm(
                    pool.imap(_read_frame_worker, work),
                    total=len(paths),
                    desc="Reading EXRs",
                )
            )

    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
    return video_tensor.unsqueeze(0)


def _save_depth_worker(args):
    output_pattern, frame_number, depth_frame = args
    output_path = format_sequence_path(output_pattern, frame_number)
    save_depth_exr(output_path, depth_frame)
    return output_path


def save_depth_sequence(depth, frame_numbers, output_pattern, workers=None):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if len(depth) != len(frame_numbers):
        raise ValueError("Depth frame count does not match input frame count.")

    if workers is None:
        workers = min(os.cpu_count() or 4, len(frame_numbers))

    work = [
        (output_pattern, frame_number, depth_frame)
        for frame_number, depth_frame in zip(frame_numbers, depth)
    ]

    if len(work) <= 2:
        outputs = [_save_depth_worker(args) for args in tqdm(work, desc="Writing EXRs")]
    else:
        with Pool(workers) as pool:
            outputs = list(
                tqdm(
                    pool.imap(_save_depth_worker, work),
                    total=len(work),
                    desc="Writing EXRs",
                )
            )
    return outputs


# =============================
# Helper: Resizing
# =============================
def resize_for_training_scale(video_tensor, target_h=480, target_w=640):
    B, T, C, H, W = video_tensor.shape
    ratio = max(target_h / H, target_w / W)
    new_H = int(np.ceil(H * ratio))
    new_W = int(np.ceil(W * ratio))

    new_H = (new_H + 15) // 16 * 16
    new_W = (new_W + 15) // 16 * 16

    if new_H == H and new_W == W:
        return video_tensor, (H, W)

    video_reshape = video_tensor.view(B * T, C, H, W)
    resized = F.interpolate(
        video_reshape,
        size=(new_H, new_W),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.view(B, T, C, new_H, new_W)
    return resized, (H, W)


def resize_depth_back(depth_np, orig_size):
    orig_H, orig_W = orig_size
    depth_tensor = torch.from_numpy(depth_np).permute(0, 3, 1, 2).float()
    depth_tensor = F.interpolate(
        depth_tensor,
        size=(orig_H, orig_W),
        mode="bilinear",
        align_corners=False,
    )
    return depth_tensor.permute(0, 2, 3, 1).cpu().numpy()


def pad_time_mod4(video_tensor):
    """Pads the temporal dimension to satisfy 4n+1 requirement."""
    B, T, C, H, W = video_tensor.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = video_tensor[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, pad_frames], dim=1)
    return video_tensor, T


def get_window_index(T, window_size, overlap):
    if T <= window_size:
        return [(0, T)]
    res = [(0, window_size)]
    start = window_size - overlap
    while start < T:
        end = start + window_size
        if end < T:
            res.append((start, end))
            start += window_size - overlap
        else:
            start = max(0, T - window_size)
            res.append((start, T))
            break
    return res


def get_trimmed_window_plan(T, window_size, overlap, trim_frames):
    """Plans fixed-length inference windows with warmup trimmed from later windows."""
    if trim_frames == 0:
        inference_windows = get_window_index(T, window_size, overlap)
        output_windows = inference_windows.copy()
        return inference_windows, output_windows

    usable_frames = window_size - trim_frames
    stride = usable_frames - overlap
    if usable_frames <= 0:
        raise ValueError(
            f"trim_frames must be smaller than window_size, got trim_frames={trim_frames}, "
            f"window_size={window_size}"
        )
    if stride <= 0:
        raise ValueError(
            "overlap must be smaller than the usable frames after trimming, got "
            f"overlap={overlap}, usable_frames={usable_frames}"
        )

    if T <= window_size:
        return [(0, T)], [(0, T)]

    inference_windows = [(0, window_size)]
    output_windows = [(0, window_size)]

    start = stride
    while start < T:
        end = start + window_size
        if end < T:
            inference_windows.append((start, end))
            output_windows.append((start + trim_frames, end))
            start += stride
        else:
            start = max(0, T - window_size)
            if inference_windows[-1][0] != start:
                inference_windows.append((start, T))
                output_windows.append((start + trim_frames, T))
            break

    return inference_windows, output_windows


# =============================
# Core Inference
# =============================
def generate_depth_sliced(
    model,
    input_rgb,
    window_size=45,
    overlap=9,
    scale_only=False,
    trim_frames=0,
):
    B, T, C, H, W = input_rgb.shape
    depth_windows, output_windows = get_trimmed_window_plan(
        T, window_size, overlap, trim_frames
    )
    print(f"depth_windows {depth_windows}")
    print(f"output_windows {output_windows}")

    depth_res_list = []

    for window_idx, (start, end) in enumerate(tqdm(depth_windows, desc="Inferencing Slices")):
        input_rgb_slice = input_rgb[:, start:end]

        input_rgb_slice, origin_T = pad_time_mod4(input_rgb_slice)
        input_frame = input_rgb_slice.shape[1]
        input_height, input_width = input_rgb_slice.shape[-2:]

        outputs = model.pipe(
            prompt=[""] * B,
            negative_prompt=[""] * B,
            mode=model.args.mode,
            height=input_height,
            width=input_width,
            num_frames=input_frame,
            batch_size=B,
            input_image=input_rgb_slice[:, 0],
            extra_images=input_rgb_slice,
            extra_image_frame_index=torch.ones([B, input_frame]).to(model.pipe.device),
            input_video=input_rgb_slice,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        depth = depth_to_single_channel(outputs["depth"])
        trimmed_start = 0 if window_idx == 0 else min(trim_frames, origin_T)
        depth_res_list.append(depth[:, trimmed_start:origin_T])

    depth_list_aligned = None
    prev_end = None

    for i, (t, (start, end)) in enumerate(zip(depth_res_list, output_windows)):
        print(f"Handling window {i} start: {start}, end: {end}")

        if i == 0:
            depth_list_aligned = t
            prev_end = end
            continue

        curr_start = start
        real_overlap = prev_end - curr_start

        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]
            curr_frames = t[:, :real_overlap]

            if scale_only:
                scale = np.sum(curr_frames * ref_frames) / (
                    np.sum(curr_frames * curr_frames) + 1e-6
                )
                shift = 0.0
            else:
                scale, shift = compute_scale_and_shift(curr_frames, ref_frames)

            scale = np.clip(scale, 0.7, 1.5)

            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0

            curr_overlap_aligned = aligned_t[:, :real_overlap]
            diff = np.abs(curr_overlap_aligned - ref_frames)
            mae_scalar = float(diff.mean(axis=tuple(range(1, diff.ndim))).mean())

            print(f"\n[Overlap {i}]")
            print(f"real_overlap = {real_overlap}")
            print(f"scale = {scale:.8f}, shift = {shift:.8f}")
            print(
                f"aligned curr range = {aligned_t.min():.6f} ~ {aligned_t.max():.6f}"
            )
            print(f"overlap MAE(after align) = {mae_scalar:.6f}")

            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(
                1, real_overlap, 1, 1, 1
            )
            smooth_overlap = (1 - alpha) * ref_frames + alpha * aligned_t[:, :real_overlap]

            depth_list_aligned = np.concatenate(
                [
                    depth_list_aligned[:, :-real_overlap],
                    smooth_overlap,
                    aligned_t[:, real_overlap:],
                ],
                axis=1,
            )
        else:
            depth_list_aligned = np.concatenate([depth_list_aligned, t], axis=1)

        print(
            "Total depth range after concat = "
            f"{depth_list_aligned.min():.6f} ~ {depth_list_aligned.max():.6f}"
        )
        prev_end = end

    return depth_list_aligned[:, :T]


# =============================
# Pipeline Components
# =============================
def load_model(ckpt_dir, yaml_args):
    accelerator = Accelerator()
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=yaml_args.lora_rank,
        lora_base_model=yaml_args.lora_base_model,
        args=yaml_args,
    )

    ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
    state_dict = load_file(ckpt_path, device="cpu")
    dit_state_dict = {
        k.replace("pipe.dit.", ""): v
        for k, v in state_dict.items()
        if "pipe.dit." in k
    }
    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    model.merge_lora_layer()
    model = model.to("cuda")

    return model


def load_exr_data(args):
    sequence = gather_exr_sequence(args.input_sequence)
    sequence = filter_frame_range(sequence, args.start, args.end)
    paths = [path for path, _ in sequence]
    frame_numbers = [frame for _, frame in sequence]

    input_tensor = read_exr_sequence(paths, width=args.width, height=args.height)
    print("Input shape:", input_tensor.shape)
    print(f"input range {input_tensor.min()} - {input_tensor.max()}")

    return input_tensor, frame_numbers


def predict_depth(model, input_tensor, args):
    depth = generate_depth_sliced(
        model,
        input_tensor,
        args.window_size,
        args.overlap,
        trim_frames=args.trim_frames,
    )[0]
    print(f"depth range shape {depth.min()} - {depth.max()}, shape {depth.shape}")

    return depth


def save_results(depth, frame_numbers, args):
    outputs = save_depth_sequence(depth, frame_numbers, args.output_sequence)
    print(f"Saved {len(outputs)} EXR frames to pattern {args.output_sequence}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpt", required=False)
    parser.add_argument("-i", "--input_sequence", type=str, required=True)
    parser.add_argument("-o", "--output_sequence", type=str, required=True)
    parser.add_argument("--model_config", default="ckpt/model_config.yaml")
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=9)
    parser.add_argument(
        "--trim_frames",
        type=int,
        default=0,
        help="Warmup frames to prepend and discard from each inference window.",
    )
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    return parser.parse_args()


# =============================
# Main Script
# =============================
def main():
    args = parse_args()
    if args.trim_frames < 0:
        raise ValueError(f"trim_frames must be non-negative, got {args.trim_frames}")
    if args.start is not None and args.end is not None and args.start > args.end:
        raise ValueError(f"Invalid frame range: start={args.start} is greater than end={args.end}")

    yaml_args = OmegaConf.load(args.model_config)

    model = load_model(args.ckpt, yaml_args)
    input_tensor, frame_numbers = load_exr_data(args)
    depth = predict_depth(model, input_tensor, args)
    save_results(depth, frame_numbers, args)

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
