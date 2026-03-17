import argparse
import gc
import glob
import os
import re
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from examples.wanvideo.model_training.WanTrainingModule import \
    WanTrainingModule
try:
    from test_script.exr import (filter_frame_range, format_sequence_path,
                                 gather_exr_sequence, linear_to_srgb,
                                 read_depth_exr, read_exr_rgb,
                                 save_rgb_depth_exr)
except ModuleNotFoundError:
    from exr import (filter_frame_range, format_sequence_path,
                     gather_exr_sequence, linear_to_srgb, read_depth_exr,
                     read_exr_rgb, save_rgb_depth_exr)


# =============================
# Helper: Math & Alignment
# =============================
def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
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


# =============================
# Helper: Windowing
# =============================
def pad_time_mod4(video_tensor):
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


def make_window_output_pattern(output_pattern, window_index):
    match = re.search(r"(%0?\d*d)", output_pattern)
    if match is None:
        raise ValueError(
            "Output path must contain a printf-style frame token such as %04d."
        )
    return (
        output_pattern[:match.start()]
        + f"_window{window_index:03d}."
        + output_pattern[match.start():]
    )


def _read_frame_worker(args):
    path, width, height = args
    frame_np = read_exr_rgb(path)
    frame_np = np.clip(linear_to_srgb(frame_np), 0.0, 1.0)

    orig_height, orig_width = frame_np.shape[:2]
    if width * height < orig_width * orig_height:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    frame_np = cv2.resize(frame_np, (width, height), interpolation=interpolation)
    return frame_np.astype(np.float32)


def load_input_window(paths, width, height, workers=None):
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
    del frames
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
    del video_np
    return video_tensor.unsqueeze(0)


def _save_depth_worker(args):
    output_pattern, frame_number, depth_frame = args
    output_path = format_sequence_path(output_pattern, frame_number)
    save_rgb_depth_exr(output_path, depth_frame)
    return output_path


def save_depth_sequence(depth, frame_numbers, output_pattern, workers=None, desc="Writing EXRs"):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if workers is None:
        workers = min(os.cpu_count() or 4, len(frame_numbers))

    work = [
        (output_pattern, frame_number, depth_frame)
        for frame_number, depth_frame in zip(frame_numbers, depth)
    ]

    if len(work) <= 2:
        outputs = [_save_depth_worker(args) for args in tqdm(work, desc=desc)]
    else:
        with Pool(workers) as pool:
            outputs = list(
                tqdm(
                    pool.imap(_save_depth_worker, work),
                    total=len(work),
                    desc=desc,
                )
            )
    return outputs


def clear_memory(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =============================
# Core Inference
# =============================
def infer_depth_window(model, input_rgb):
    with torch.inference_mode():
        input_rgb_slice, origin_T = pad_time_mod4(input_rgb)
        input_frame = input_rgb_slice.shape[1]
        input_height, input_width = input_rgb_slice.shape[-2:]

        outputs = model.pipe(
            prompt=[""],
            negative_prompt=[""],
            mode=model.args.mode,
            height=input_height,
            width=input_width,
            num_frames=input_frame,
            batch_size=1,
            input_image=input_rgb_slice[:, 0],
            extra_images=input_rgb_slice,
            extra_image_frame_index=torch.ones([1, input_frame]).to(model.pipe.device),
            input_video=input_rgb_slice,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        depth = depth_to_single_channel(outputs["depth"])[:, :origin_T]

    depth_np = depth[0]
    if isinstance(depth_np, torch.Tensor):
        depth_np = depth_np.detach().float().cpu().numpy()
    return depth_np


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
    model.pipe.dit.eval()
    return model


def run_window_inference(model, sequence, inference_windows, output_windows, args):
    for window_index, ((start, end), (out_start, out_end)) in enumerate(
        zip(inference_windows, output_windows)
    ):
        window_frames = sequence[start:end]
        output_frames = sequence[out_start:out_end]
        paths = [path for path, _ in window_frames]
        frame_numbers = [frame for _, frame in window_frames]
        output_frame_numbers = [frame for _, frame in output_frames]
        output_pattern = make_window_output_pattern(args.output_sequence, window_index)

        print(
            f"Window {window_index:03d}: source frames {frame_numbers[0]}-{frame_numbers[-1]} "
            f"({len(frame_numbers)} frames)"
        )

        input_tensor = load_input_window(paths, args.width, args.height)
        print(f"Window input shape: {input_tensor.shape}")
        print(f"Window input range: {input_tensor.min()} - {input_tensor.max()}")

        depth = infer_depth_window(model, input_tensor)
        trimmed_start = 0 if window_index == 0 else min(args.trim_frames, len(frame_numbers))
        depth = depth[trimmed_start:len(frame_numbers)]
        print(f"Window depth range: {depth.min()} - {depth.max()}, shape {depth.shape}")
        save_depth_sequence(
            depth,
            output_frame_numbers,
            output_pattern,
            desc=f"Writing window {window_index:03d}",
        )

        clear_memory(input_tensor, depth)


def unload_model(model):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def read_window_depths(output_pattern, frame_numbers):
    frames = [
        read_depth_exr(format_sequence_path(output_pattern, frame_number))
        for frame_number in frame_numbers
    ]
    return np.stack(frames, axis=0).astype(np.float32)


def write_final_chunk(output_pattern, frame_numbers, chunk):
    save_depth_sequence(chunk, frame_numbers, output_pattern, desc="Writing final EXRs")


def align_windowed_depth(sequence, windows, args):
    next_overlap = 0
    pending_frames = None
    pending_frame_numbers = None

    for window_index, (start, end) in enumerate(windows):
        window_frames = sequence[start:end]
        frame_numbers = [frame for _, frame in window_frames]
        window_pattern = make_window_output_pattern(args.output_sequence, window_index)
        window_depth = read_window_depths(window_pattern, frame_numbers)

        if window_index == 0:
            if len(windows) > 1:
                next_start = windows[1][0]
                next_overlap = end - next_start
            else:
                next_overlap = 0

            if next_overlap > 0:
                prefix_count = len(frame_numbers) - next_overlap
                if prefix_count > 0:
                    write_final_chunk(
                        args.output_sequence,
                        frame_numbers[:prefix_count],
                        window_depth[:prefix_count],
                    )
                pending_frames = window_depth[prefix_count:].copy()
                pending_frame_numbers = frame_numbers[prefix_count:]
            else:
                write_final_chunk(args.output_sequence, frame_numbers, window_depth)
                pending_frames = None
                pending_frame_numbers = None

            clear_memory(window_depth)
            continue

        real_overlap = len(pending_frame_numbers) if pending_frame_numbers is not None else 0
        if real_overlap > 0:
            ref_frames = pending_frames
            curr_frames = window_depth[:real_overlap]
            scale, shift = compute_scale_and_shift(curr_frames, ref_frames)
            scale = np.clip(scale, 0.7, 1.5)

            aligned_window = window_depth * scale + shift
            aligned_window[aligned_window < 0] = 0

            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(
                real_overlap, 1, 1
            )
            smooth_overlap = (1 - alpha) * ref_frames + alpha * aligned_window[:real_overlap]

            print(
                f"Align window {window_index:03d}: overlap={real_overlap}, "
                f"scale={scale:.8f}, shift={shift:.8f}"
            )
        else:
            aligned_window = window_depth
            smooth_overlap = None

        if window_index < len(windows) - 1:
            next_start = windows[window_index + 1][0]
            next_overlap = end - next_start
        else:
            next_overlap = 0

        write_frames = []
        write_numbers = []

        if real_overlap > 0:
            write_frames.append(smooth_overlap)
            write_numbers.extend(frame_numbers[:real_overlap])

        unique_tail = aligned_window[real_overlap:]
        unique_tail_numbers = frame_numbers[real_overlap:]

        if next_overlap > 0:
            split_index = len(unique_tail_numbers) - next_overlap
            if split_index > 0:
                write_frames.append(unique_tail[:split_index])
                write_numbers.extend(unique_tail_numbers[:split_index])
            pending_frames = unique_tail[split_index:].copy()
            pending_frame_numbers = unique_tail_numbers[split_index:]
        else:
            if len(unique_tail_numbers) > 0:
                write_frames.append(unique_tail)
                write_numbers.extend(unique_tail_numbers)
            pending_frames = None
            pending_frame_numbers = None

        if write_frames:
            chunk = np.concatenate(write_frames, axis=0)
            write_final_chunk(args.output_sequence, write_numbers, chunk)
            clear_memory(chunk)

        clear_memory(window_depth, aligned_window)

    if pending_frames is not None and pending_frame_numbers:
        write_final_chunk(args.output_sequence, pending_frame_numbers, pending_frames)
        clear_memory(pending_frames)


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
        default=16,
        help="Warmup frames to discard from each inference window after the first.",
    )
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trim_frames < 0:
        raise ValueError(f"trim_frames must be non-negative, got {args.trim_frames}")
    if args.start is not None and args.end is not None and args.start > args.end:
        raise ValueError(f"Invalid frame range: start={args.start} is greater than end={args.end}")

    sequence = gather_exr_sequence(args.input_sequence)
    sequence = filter_frame_range(sequence, args.start, args.end)
    inference_windows, output_windows = get_trimmed_window_plan(
        len(sequence), args.window_size, args.overlap, args.trim_frames
    )
    print(
        f"Found {len(sequence)} frames across {len(inference_windows)} windows: "
        f"inference={inference_windows}, output={output_windows}"
    )

    yaml_args = OmegaConf.load(args.model_config)
    model = load_model(args.ckpt, yaml_args)

    run_window_inference(model, sequence, inference_windows, output_windows, args)
    unload_model(model)

    align_windowed_depth(sequence, output_windows, args)
    print("Chunked inference completed successfully!")


if __name__ == "__main__":
    main()
