import argparse
import gc
import os
import random
import signal
import sys
import time
from datetime import timedelta
from itertools import cycle
from pathlib import Path, PurePosixPath

import csv
import numpy as np

_cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _cuda_alloc_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        f"{_cuda_alloc_conf},expandable_segments:True"
        if _cuda_alloc_conf
        else "expandable_segments:True"
    )

import torch
import torch.nn.functional as F
from accelerate import Accelerator, accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import Dataset
from tqdm import tqdm

from examples.dataset import (HypersimDataset, KITTI_VID_Dataset, NYUv2Dataset,
                              Scannet_VID_Dataset, TartanAir_VID_Dataset,
                              VKITTI_VID_Dataset, VKITTIDataset)
from examples.dataset.hypersim_dataset import (HypersimImageDepthNormalTransform,
                                               aligned_resolution_candidates,
                                               sample_resolution)
# Import modules
from examples.wanvideo.model_training.DiffusionTrainingModule import \
    DiffusionTrainingModule
from examples.wanvideo.model_training.WanTrainingModule import (
    Validation, WanTrainingModule)

from .training_loss import GradientLoss3DSeparate

try:
    import Imath
    import OpenEXR
except ImportError:
    Imath = None
    OpenEXR = None

process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))


os.environ["TOKENIZERS_PARALLELISM"] = "false"


_interrupt_requested = False
_interrupt_count = 0
_last_known_global_step = 0


class _TeeStream:
    """Write to both a terminal stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        try:
            # tqdm redraws the active progress line with carriage returns.
            # Mirroring those raw fragments into the log file produces
            # truncated lines like `global_stMicrostep ...`.
            if "\r" in data and "\n" not in data:
                return
            self._log_file.write(data.replace("\r", "\n"))
        except Exception:
            pass

    def flush(self):
        self._original.flush()
        try:
            self._log_file.flush()
        except Exception:
            pass

    def isatty(self):
        return self._original.isatty()

    def fileno(self):
        return self._original.fileno()


def load_synthhuman_depth_exr(depth_path):
    if OpenEXR is None or Imath is None:
        raise ImportError(
            "SynthHumanDataset requires the OpenEXR Python package "
            "(imports `OpenEXR` and `Imath`)."
        )

    exr_file = OpenEXR.InputFile(depth_path)
    try:
        header = exr_file.header()
        data_window = header["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        channel_map = header["channels"]
        channel_names = list(channel_map.keys())
        preferred = ("Z", "Y", "R", "G", "B")
        channel_name = next(
            (name for name in preferred if name in channel_names),
            sorted(channel_names)[0],
        )
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = exr_file.channel(channel_name, pixel_type)
        depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        return depth
    finally:
        exr_file.close()


def _progress_log(message):
    tqdm.write(message, file=sys.stdout)
    sys.stdout.flush()


def _bytes_to_gib(num_bytes):
    return num_bytes / (1024 ** 3)


def _reset_cuda_peak_memory_stats(accelerator):
    device = accelerator.device
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    torch.cuda.reset_peak_memory_stats(device_index)


def _empty_cuda_cache(accelerator):
    device = accelerator.device
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()


def _log_cuda_memory(accelerator, label):
    device = accelerator.device
    if device.type != "cuda" or not torch.cuda.is_available():
        _progress_log(
            f"GPU {accelerator.process_index} CUDA memory [{label}] unavailable on device {device}."
        )
        return

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    allocated_bytes = torch.cuda.memory_allocated(device_index)
    reserved_bytes = torch.cuda.memory_reserved(device_index)
    max_allocated_bytes = torch.cuda.max_memory_allocated(device_index)
    max_reserved_bytes = torch.cuda.max_memory_reserved(device_index)
    _progress_log(
        f"GPU {accelerator.process_index} CUDA memory [{label}] "
        f"allocated={_bytes_to_gib(allocated_bytes):.2f} GiB, "
        f"reserved={_bytes_to_gib(reserved_bytes):.2f} GiB, "
        f"free={_bytes_to_gib(free_bytes):.2f} GiB, "
        f"total={_bytes_to_gib(total_bytes):.2f} GiB, "
        f"peak_allocated={_bytes_to_gib(max_allocated_bytes):.2f} GiB, "
        f"peak_reserved={_bytes_to_gib(max_reserved_bytes):.2f} GiB"
    )


def _wan_latent_tokens(num_frames, height, width):
    latent_frames = (int(num_frames) - 1) // 4 + 1
    return latent_frames * (int(height) // 16) * (int(width) // 16)


def sanitize_infinigen_depth(depth):
    finite_positive = np.isfinite(depth) & (depth > 0)
    if not finite_positive.any():
        raise ValueError("depth has no finite positive values")
    sanitized = depth.astype(np.float32, copy=True)
    # Infinigen commonly uses inf background; replace it with the farthest
    # finite depth so the existing normalization path can run without masks.
    sanitized[~finite_positive] = np.max(sanitized[finite_positive])
    return sanitized


def _normalize_manifest_entry(entry):
    normalized = entry.strip().replace("\\", "/").strip("/")
    if normalized in ("", "."):
        return None
    return normalized


def _load_split_manifest(manifest_path):
    if manifest_path is None:
        return None

    manifest_path = str(manifest_path).strip()
    if not manifest_path:
        return None
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Split manifest does not exist: {manifest_path}"
        )

    entries = set()
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0]
            normalized = _normalize_manifest_entry(line)
            if normalized is not None:
                entries.add(normalized)

    if not entries:
        raise ValueError(f"Split manifest is empty: {manifest_path}")

    return entries


def _relative_posix(path, root):
    relative = path.relative_to(root).as_posix()
    return "" if relative == "." else relative


def _candidate_matches_manifest(candidate, manifest_entries):
    for entry in manifest_entries:
        if candidate == entry or candidate.startswith(entry + "/"):
            return True
    return False


def _infinigen_manifest_candidates(image_path, depth_path, data_root):
    candidates = {
        _relative_posix(image_path, data_root),
        _relative_posix(depth_path, data_root),
        _relative_posix(image_path.parent, data_root),
        _relative_posix(depth_path.parent, data_root),
    }

    for rel_parts, marker in (
        (image_path.relative_to(data_root).parts, "Image"),
        (depth_path.relative_to(data_root).parts, "Depth"),
    ):
        if marker not in rel_parts:
            continue
        marker_idx = rel_parts.index(marker)
        if marker_idx > 0:
            candidates.add(PurePosixPath(*rel_parts[:marker_idx]).as_posix())
        candidates.add(PurePosixPath(*rel_parts[: marker_idx + 1]).as_posix())

    candidates.discard("")
    return candidates


def _infinigen_sample_in_manifest(image_path, depth_path, data_root, manifest_entries):
    candidates = _infinigen_manifest_candidates(
        image_path=image_path,
        depth_path=depth_path,
        data_root=data_root,
    )
    return any(
        _candidate_matches_manifest(candidate, manifest_entries)
        for candidate in candidates
    )


def _resolve_infinigen_depth_path(image_path):
    depth_name = image_path.name.replace("Image", "Depth", 1).rsplit(".", 1)[0] + ".npy"
    if "/Image/" in str(image_path):
        depth_path = Path(str(image_path).replace("/Image/", "/Depth/")).with_name(depth_name)
    else:
        depth_path = image_path.with_name(depth_name)
    if depth_path.exists():
        return depth_path
    return None


def _infer_infinigen_sequence_key(image_path, data_root):
    rel_parts = image_path.relative_to(data_root).parts
    if "Image" in rel_parts:
        image_idx = rel_parts.index("Image")
        if image_idx > 0:
            return PurePosixPath(*rel_parts[:image_idx]).as_posix()
        return "Image"

    parent = image_path.parent.relative_to(data_root).as_posix()
    return "." if parent == "." else parent


class SynthHumanDataset(Dataset):
    def __init__(
        self,
        data_dir,
        random_flip,
        norm_type,
        resolution=(480, 640),
        truncnorm_min=0.02,
        start=0,
        train_ratio=1.0,
        min_resolution=None,
        max_resolution=None,
    ):
        self.data_dir = data_dir
        self.data_list = []
        self.invalid_indices = set()
        self.reported_invalid_indices = set()

        if OpenEXR is None or Imath is None:
            raise ImportError(
                "SynthHumanDataset requires the OpenEXR Python package "
                "(imports `OpenEXR` and `Imath`)."
            )

        for root, _, files in os.walk(data_dir):
            for filename in files:
                if not filename.startswith("rgb_") or not filename.endswith(".png"):
                    continue
                rgb_path = os.path.join(root, filename)
                depth_name = filename.replace("rgb_", "depth_", 1).rsplit(".", 1)[0] + ".exr"
                depth_path = os.path.join(root, depth_name)
                if os.path.exists(depth_path):
                    self.data_list.append((rgb_path, depth_path))

        self.data_list.sort()
        self.data_list = self.data_list[start:]
        if not self.data_list:
            raise RuntimeError(
                f"No SynthHuman RGB/depth pairs found under {data_dir}"
            )

        new_h, new_w = resolution
        self.new_h = new_h
        self.new_w = new_w
        self.min_resolution = tuple(min_resolution) if min_resolution else None
        self.max_resolution = tuple(max_resolution) if max_resolution else None
        self.transform = HypersimImageDepthNormalTransform(
            (new_h, new_w), random_flip, norm_type, truncnorm_min, False
        )

        if train_ratio < 1.0:
            origin_len = len(self.data_list)
            self.data_list = self.data_list[:int(origin_len * train_ratio)]
            print(
                f"SynthHuman use {int(origin_len * train_ratio)} samples instead of {origin_len}..."
            )
        else:
            print(f"SynthHuman use origin {len(self.data_list)} samples...")

    def __len__(self):
        return len(self.data_list)

    def _report_invalid_sample(self, idx, reason, img_path, dep_path):
        if idx in self.reported_invalid_indices:
            return
        self.reported_invalid_indices.add(idx)
        print(
            f"Skipping invalid SynthHuman sample at index {idx}: {reason}. "
            f"image={img_path}, depth={dep_path}"
        )

    def _next_valid_index(self, idx):
        if len(self.invalid_indices) >= len(self.data_list):
            raise RuntimeError("All SynthHuman samples are invalid.")
        next_idx = (idx + 1) % len(self.data_list)
        while next_idx in self.invalid_indices:
            next_idx = (next_idx + 1) % len(self.data_list)
        return next_idx

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        if idx in self.invalid_indices:
            return self.__getitem__(self._next_valid_index(idx))

        try:
            img_path, dep_path = self.data_list[idx]

            if self.min_resolution and self.max_resolution:
                res = sample_resolution(self.min_resolution, self.max_resolution)
            else:
                res = (self.new_h, self.new_w)

            image = Image.open(img_path).convert("RGB")

            # SynthHuman depth is published in centimeters; convert to meters.
            depth = load_synthhuman_depth_exr(dep_path) / 100.0
            if not np.isfinite(depth).all():
                raise ValueError("depth contains NaN or inf")
            if (depth <= 0).any():
                raise ValueError("depth contains non-positive values")

            raw_depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            raw_depth = F.interpolate(
                raw_depth, size=res, mode="nearest"
            ).squeeze()
            raw_depth = torch.clamp(raw_depth, 1e-3, 65).repeat(3, 1, 1)

            image, depth, normal = self.transform(image, depth, None, size=res)
            if torch.isnan(image).any() or torch.isinf(image).any():
                raise ValueError("image is nan or inf after transform")
            if torch.isnan(depth).any() or torch.isinf(depth).any():
                raise ValueError("depth is nan or inf after transform")

            return {
                "sample_idx": torch.tensor(idx),
                "images": image.unsqueeze(0),
                "disparity": depth.unsqueeze(0),
                "depth_raw_linear": raw_depth.unsqueeze(0),
                "depth": raw_depth.unsqueeze(0),
                "normal_values": normal,
                "image_path": img_path,
                "depth_path": dep_path,
            }
        except Exception as exc:
            img_path = self.data_list[idx][0] if self.data_list else "unknown"
            dep_path = self.data_list[idx][1] if self.data_list else "unknown"
            self.invalid_indices.add(idx)
            self._report_invalid_sample(idx, str(exc), img_path, dep_path)
            return self.__getitem__(self._next_valid_index(idx))


class InfinigenDataset(Dataset):
    def __init__(
        self,
        data_dir,
        random_flip,
        norm_type,
        resolution=(480, 640),
        truncnorm_min=0.02,
        start=0,
        train_ratio=1.0,
        split_manifest=None,
        min_resolution=None,
        max_resolution=None,
    ):
        self.data_dir = data_dir
        self.data_list = []
        self.invalid_indices = set()
        self.reported_invalid_indices = set()
        data_root = Path(data_dir)
        manifest_entries = _load_split_manifest(split_manifest)
        total_pairs = 0

        for image_path in sorted(data_root.rglob("Image*.png")):
            depth_name = image_path.name.replace("Image", "Depth", 1).rsplit(".", 1)[0] + ".npy"
            if "/Image/" in str(image_path):
                depth_path = Path(str(image_path).replace("/Image/", "/Depth/")).with_name(depth_name)
            else:
                depth_path = image_path.with_name(depth_name)
            if depth_path.exists():
                total_pairs += 1
                if manifest_entries is not None and not _infinigen_sample_in_manifest(
                    image_path=image_path,
                    depth_path=depth_path,
                    data_root=data_root,
                    manifest_entries=manifest_entries,
                ):
                    continue
                self.data_list.append((str(image_path), str(depth_path)))

        self.data_list = self.data_list[start:]
        if not self.data_list:
            raise RuntimeError(
                f"No Infinigen RGB/depth pairs found under {data_dir}"
            )
        if manifest_entries is not None:
            print(
                f"Infinigen manifest {split_manifest} kept "
                f"{len(self.data_list)} of {total_pairs} RGB/depth pairs."
            )

        new_h, new_w = resolution
        self.new_h = new_h
        self.new_w = new_w
        self.min_resolution = tuple(min_resolution) if min_resolution else None
        self.max_resolution = tuple(max_resolution) if max_resolution else None
        self.transform = HypersimImageDepthNormalTransform(
            (new_h, new_w), random_flip, norm_type, truncnorm_min, False
        )

        if train_ratio < 1.0:
            origin_len = len(self.data_list)
            self.data_list = self.data_list[:int(origin_len * train_ratio)]
            print(
                f"Infinigen use {int(origin_len * train_ratio)} samples instead of {origin_len}..."
            )
        else:
            print(f"Infinigen use origin {len(self.data_list)} samples...")

    def __len__(self):
        return len(self.data_list)

    def _report_invalid_sample(self, idx, reason, img_path, dep_path):
        if idx in self.reported_invalid_indices:
            return
        self.reported_invalid_indices.add(idx)
        print(
            f"Skipping invalid Infinigen sample at index {idx}: {reason}. "
            f"image={img_path}, depth={dep_path}"
        )

    def _next_valid_index(self, idx):
        if len(self.invalid_indices) >= len(self.data_list):
            raise RuntimeError("All Infinigen samples are invalid.")
        next_idx = (idx + 1) % len(self.data_list)
        while next_idx in self.invalid_indices:
            next_idx = (next_idx + 1) % len(self.data_list)
        return next_idx

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        if idx in self.invalid_indices:
            return self.__getitem__(self._next_valid_index(idx))

        try:
            img_path, dep_path = self.data_list[idx]

            if self.min_resolution and self.max_resolution:
                res = sample_resolution(self.min_resolution, self.max_resolution)
            else:
                res = (self.new_h, self.new_w)

            image = Image.open(img_path).convert("RGB")

            depth = np.load(dep_path)
            if depth.ndim != 2:
                raise ValueError(f"expected HxW depth, got shape {depth.shape}")
            depth = sanitize_infinigen_depth(depth)

            raw_depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            raw_depth = F.interpolate(
                raw_depth, size=res, mode="nearest"
            ).squeeze()
            raw_depth = torch.clamp(raw_depth, 1e-3, 65).repeat(3, 1, 1)

            image, depth, normal = self.transform(image, depth, None, size=res)
            if torch.isnan(image).any() or torch.isinf(image).any():
                raise ValueError("image is nan or inf after transform")
            if torch.isnan(depth).any() or torch.isinf(depth).any():
                raise ValueError("depth is nan or inf after transform")

            return {
                "sample_idx": torch.tensor(idx),
                "images": image.unsqueeze(0),
                "disparity": depth.unsqueeze(0),
                "depth_raw_linear": raw_depth.unsqueeze(0),
                "depth": raw_depth.unsqueeze(0),
                "normal_values": normal,
                "image_path": img_path,
                "depth_path": dep_path,
            }
        except Exception as exc:
            img_path = self.data_list[idx][0] if self.data_list else "unknown"
            dep_path = self.data_list[idx][1] if self.data_list else "unknown"
            self.invalid_indices.add(idx)
            self._report_invalid_sample(idx, str(exc), img_path, dep_path)
            return self.__getitem__(self._next_valid_index(idx))


class InfinigenVideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        random_flip,
        norm_type,
        resolution=(480, 640),
        truncnorm_min=0.02,
        max_num_frame=None,
        min_num_frame=None,
        max_sample_stride=None,
        min_sample_stride=None,
        split_manifest=None,
        deterministic_sampling=False,
        min_resolution=None,
        max_resolution=None,
        resolution_budget_num_frames=None,
        resolution_budget_scale=1.0,
        log_sample_shapes=0,
    ):
        self.data_dir = data_dir
        self.data_list = []
        self.invalid_indices = set()
        self.reported_invalid_indices = set()
        self.deterministic_sampling = deterministic_sampling
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.max_sample_stride = max_sample_stride
        self.min_sample_stride = min_sample_stride
        self.num_frames = list(range(min_num_frame, max_num_frame + 1))
        self.strides = list(range(min_sample_stride, max_sample_stride + 1))
        valid_frame_counts = [frame_count for frame_count in self.num_frames if frame_count % 4 == 1]
        if not valid_frame_counts:
            raise ValueError(
                "InfinigenVideoDataset requires at least one frame count where num_frames % 4 == 1."
            )

        data_root = Path(data_dir)
        manifest_entries = _load_split_manifest(split_manifest)
        total_pairs = 0
        sequence_map = {}

        for image_path in sorted(data_root.rglob("Image*.png")):
            depth_path = _resolve_infinigen_depth_path(image_path)
            if depth_path is None:
                continue
            total_pairs += 1
            if manifest_entries is not None and not _infinigen_sample_in_manifest(
                image_path=image_path,
                depth_path=depth_path,
                data_root=data_root,
                manifest_entries=manifest_entries,
            ):
                continue

            sequence_key = _infer_infinigen_sequence_key(image_path, data_root)
            record = sequence_map.setdefault(
                sequence_key,
                {
                    "sequence_name": sequence_key,
                    "img_path_list": [],
                    "depth_path_list": [],
                },
            )
            record["img_path_list"].append(str(image_path))
            record["depth_path_list"].append(str(depth_path))

        min_required_frames = min(valid_frame_counts)
        self.data_list = [
            record
            for _, record in sorted(sequence_map.items())
            if len(record["img_path_list"]) >= min_required_frames
        ]

        if not self.data_list:
            raise RuntimeError(
                f"No Infinigen video sequences with at least {min_required_frames} frames found under {data_dir}"
            )

        if manifest_entries is not None:
            kept_pairs = sum(len(record["img_path_list"]) for record in self.data_list)
            print(
                f"Infinigen video manifest {split_manifest} kept "
                f"{kept_pairs} of {total_pairs} RGB/depth pairs across {len(self.data_list)} sequences."
            )
        else:
            print(
                f"InfinigenVideoDataset discovered {len(self.data_list)} sequences under {data_dir}."
            )

        new_h, new_w = resolution
        self.new_h = new_h
        self.new_w = new_w
        self.min_resolution = tuple(min_resolution) if min_resolution else None
        self.max_resolution = tuple(max_resolution) if max_resolution else None
        self.resolution_budget_num_frames = (
            int(resolution_budget_num_frames)
            if resolution_budget_num_frames is not None
            else None
        )
        self.resolution_budget_scale = float(resolution_budget_scale)
        self.video_pixel_budget = (
            int(
                self.new_h
                * self.new_w
                * self.resolution_budget_num_frames
                * self.resolution_budget_scale
            )
            if self.resolution_budget_num_frames is not None
            else None
        )
        self.log_sample_shapes = log_sample_shapes
        self._logged_sample_shapes = 0
        self.transform = HypersimImageDepthNormalTransform(
            (new_h, new_w), random_flip, norm_type, truncnorm_min, False
        )

    def __len__(self):
        return len(self.data_list)

    def _report_invalid_sample(self, idx, reason, sequence_name):
        if idx in self.reported_invalid_indices:
            return
        self.reported_invalid_indices.add(idx)
        print(
            f"Skipping invalid Infinigen video sample at index {idx}: {reason}. "
            f"sequence={sequence_name}"
        )

    def _next_valid_index(self, idx):
        if len(self.invalid_indices) >= len(self.data_list):
            raise RuntimeError("All Infinigen video sequences are invalid.")
        next_idx = (idx + 1) % len(self.data_list)
        while next_idx in self.invalid_indices:
            next_idx = (next_idx + 1) % len(self.data_list)
        return next_idx

    def _max_area_for_num_frames(self, num_frames):
        if self.video_pixel_budget is None:
            return None
        base_area = self.new_h * self.new_w
        scaled_base_area = int(base_area * self.resolution_budget_scale)
        budget_frames = max(int(self.resolution_budget_num_frames), 1)
        spatial_area_budget = scaled_base_area
        volume_area_budget = self.video_pixel_budget // max(num_frames, 1)
        temporal_area_budget = (
            scaled_base_area * budget_frames * budget_frames
        ) // max(num_frames * num_frames, 1)
        return min(spatial_area_budget, volume_area_budget, temporal_area_budget)

    def _select_clip_spec(self, total_frames):
        valid_choices = []
        for stride in self.strides:
            for num_frames in self.num_frames:
                if num_frames % 4 != 1:
                    continue
                total_frames_required = stride * (num_frames - 1) + 1
                if total_frames_required <= total_frames:
                    max_area = self._max_area_for_num_frames(num_frames)
                    if self.min_resolution and self.max_resolution:
                        res_candidates = aligned_resolution_candidates(
                            self.min_resolution,
                            self.max_resolution,
                            max_area=max_area,
                        )
                        if not res_candidates:
                            continue
                    elif max_area is not None and (self.new_h * self.new_w) > max_area:
                        continue
                    valid_choices.append(
                        (num_frames, stride, total_frames_required, max_area)
                    )

        if not valid_choices:
            raise ValueError(
                f"Unable to sample a valid clip from {total_frames} frames "
                f"with num_frame range {self.min_num_frame}-{self.max_num_frame} "
                f"and stride range {self.min_sample_stride}-{self.max_sample_stride}."
            )

        if self.deterministic_sampling:
            valid_choices.sort(key=lambda item: (-item[0], item[1]))
            num_frames, stride, total_frames_required, max_area = valid_choices[0]
            start_idx = max(0, (total_frames - total_frames_required) // 2)
        else:
            num_frames, stride, total_frames_required, max_area = random.choice(valid_choices)
            start_idx = random.randint(0, total_frames - total_frames_required)

        end_idx = start_idx + total_frames_required
        return list(range(start_idx, end_idx, stride)), num_frames, max_area

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        if idx in self.invalid_indices:
            return self.__getitem__(self._next_valid_index(idx))

        try:
            sample = self.data_list[idx]
            sequence_name = sample["sequence_name"]
            img_path_list = sample["img_path_list"]
            depth_path_list = sample["depth_path_list"]
            assert len(img_path_list) == len(depth_path_list)

            clip_indices, sampled_num_frames, sampled_max_area = self._select_clip_spec(
                len(img_path_list)
            )
            sampled_img_paths = [img_path_list[i] for i in clip_indices]
            sampled_depth_paths = [depth_path_list[i] for i in clip_indices]

            if self.min_resolution and self.max_resolution:
                res = sample_resolution(
                    self.min_resolution,
                    self.max_resolution,
                    max_area=sampled_max_area,
                    fallback_to_smallest=False,
                )
            else:
                res = (self.new_h, self.new_w)
            if self._logged_sample_shapes < self.log_sample_shapes:
                self._logged_sample_shapes += 1
                print(
                    "Infinigen sample "
                    f"frames={sampled_num_frames} resolution={res} "
                    f"max_area={sampled_max_area} sequence={sequence_name}"
                )

            image_list = []
            disparity_list = []
            raw_depth_list = []
            for img_path, dep_path in zip(sampled_img_paths, sampled_depth_paths):
                image = Image.open(img_path).convert("RGB")
                depth = np.load(dep_path)
                if depth.ndim != 2:
                    raise ValueError(f"expected HxW depth, got shape {depth.shape}")
                depth = sanitize_infinigen_depth(depth)

                raw_depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
                raw_depth = F.interpolate(
                    raw_depth, size=res, mode="nearest"
                ).squeeze()
                raw_depth = torch.clamp(raw_depth, 1e-3, 65).repeat(3, 1, 1)

                image, disparity, _ = self.transform(image, depth, None, size=res)
                if torch.isnan(image).any() or torch.isinf(image).any():
                    raise ValueError("image is nan or inf after transform")
                if torch.isnan(disparity).any() or torch.isinf(disparity).any():
                    raise ValueError("depth is nan or inf after transform")

                image_list.append(image)
                disparity_list.append(disparity)
                raw_depth_list.append(raw_depth)

            return {
                "sample_idx": torch.tensor(idx),
                "images": torch.stack(image_list),
                "disparity": torch.stack(disparity_list),
                "depth_raw_linear": torch.stack(raw_depth_list),
                "image_path": sampled_img_paths[0],
                "depth_path": sampled_depth_paths[0],
                "scene_name": sequence_name,
            }
        except Exception as exc:
            sequence_name = self.data_list[idx]["sequence_name"] if self.data_list else "unknown"
            self.invalid_indices.add(idx)
            self._report_invalid_sample(idx, str(exc), sequence_name)
            return self.__getitem__(self._next_valid_index(idx))


def _sigint_handler(signum, frame):
    del frame
    global _interrupt_requested, _interrupt_count
    _interrupt_count += 1
    if _interrupt_count >= 2:
        raise KeyboardInterrupt
    _interrupt_requested = True
    sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    try:
        print(
            f"\n{sig_name} received. Training will pause at the next safe point."
            " Press Ctrl+C again to exit immediately.",
            file=sys.stderr,
            flush=True,
        )
    except (BrokenPipeError, OSError):
        pass


def _save_checkpoint_on_interrupt(
    accelerator,
    model,
    model_logger,
    global_step,
    args,
    prompt_user=True,
):
    decision_path = os.path.join(model_logger.output_path, ".sigint_decision")

    if accelerator.is_main_process:
        decision = "save"
        if prompt_user and sys.stdin is not None and sys.stdin.isatty():
            try:
                while True:
                    answer = input(
                        f"\nCtrl+C received at global step {global_step}. "
                        "Save checkpoint before exiting? [y/N]: "
                    ).strip().lower()
                    if answer in ("", "n", "no"):
                        decision = "exit"
                        break
                    if answer in ("y", "yes"):
                        decision = "save"
                        break
                    print("Please answer 'y' or 'n'.", flush=True)
            except (OSError, EOFError):
                # stdin lost (e.g. parent process died) — save by default.
                print(
                    f"\nCtrl+C received at global step {global_step}, but stdin "
                    "is gone. Saving a checkpoint before exit.",
                    flush=True,
                )
                decision = "save"
        elif prompt_user:
            print(
                f"\nCtrl+C received at global step {global_step}, but no interactive "
                "stdin is available. Saving a checkpoint before exit.",
                flush=True,
            )
        with open(decision_path, "w") as f:
            f.write(decision)

    while not os.path.exists(decision_path):
        time.sleep(0.1)

    with open(decision_path, "r") as f:
        decision = f.read().strip()

    accelerator.wait_for_everyone()

    if decision == "save":
        model.pipe.dit.eval()
        print(f"GPU {accelerator.process_index} saving training state before exit...")
        accelerator.save_state(
            os.path.join(model_logger.output_path, f"checkpoint-step-{global_step}")
        )
        if accelerator.is_main_process:
            torch.save(
                {"global_step": global_step},
                os.path.join(model_logger.output_path, "trainer_state.pt"),
            )
            print(f"Checkpoint saved at step {global_step}", flush=True)
    else:
        print(
            f"GPU {accelerator.process_index} exiting without saving at step {global_step}",
            flush=True,
        )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process and os.path.exists(decision_path):
        os.remove(decision_path)

    if decision == "save":
        model.pipe.scheduler.set_timesteps(
            training=True,
            denoise_step=args.denoise_step,
        )
        model.pipe.dit.train()

    return decision == "save"


def maybe_save_and_exit_on_interrupt(accelerator, model, model_logger, global_step, args):
    global _interrupt_requested, _interrupt_count
    if not _interrupt_requested:
        return False, global_step

    # Reset count so a Ctrl+C during the save prompt doesn't immediately kill.
    _interrupt_count = 0
    _save_checkpoint_on_interrupt(
        accelerator=accelerator,
        model=model,
        model_logger=model_logger,
        global_step=global_step,
        args=args,
    )
    _interrupt_requested = False
    return True, global_step


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], str):
            collated[key] = values
        elif values[0] is None:
            collated[key] = None
        else:
            raise TypeError(
                f"Unsupported type for key '{key}': {type(values[0])}")
    return collated


def _build_tartanair_video_dataloader(args, accelerator):
    min_res = list(args.get("min_resolution", [])) or None
    max_res = list(args.get("max_resolution", [])) or None
    resolution_budget_num_frames = args.get("video_resolution_budget_num_frames")
    resolution_budget_scale = float(args.get("video_resolution_budget_scale", 1.0))
    video_num_workers = int(args.get("video_dataloader_num_workers", 1))
    video_persistent_workers = bool(video_num_workers > 0)
    dataset = TartanAir_VID_Dataset(
        data_dir=args.train_data_dir_ttr_vid,
        random_flip=args.random_flip,
        norm_type=args.norm_type,
        max_num_frame=args.max_num_frame,
        min_num_frame=args.min_num_frame,
        max_sample_stride=args.max_sample_stride,
        min_sample_stride=args.min_sample_stride,
        train_ratio=args.train_ratio,
        min_resolution=min_res,
        max_resolution=max_res,
        resolution_budget_num_frames=resolution_budget_num_frames,
        resolution_budget_scale=resolution_budget_scale,
        log_sample_shapes=int(args.get("log_video_sample_shapes", 4)),
    )
    dataset.data_list = dataset.data_list * 100
    accelerator.print(f"Enlarged length of tartanair_video: {len(dataset)}")
    dataloader_kwargs = dict(
        dataset=dataset,
        shuffle=True,
        batch_size=1,
        num_workers=video_num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=video_persistent_workers,
        drop_last=True,
    )
    if video_num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 1
    return torch.utils.data.DataLoader(**dataloader_kwargs)


def _build_vkitti_video_dataloader(args, accelerator):
    dataset = VKITTI_VID_Dataset(
        root_dir=args.train_data_dir_vkitti_vid,
        norm_type=args.norm_type,
        max_num_frame=args.max_num_frame,
        min_num_frame=args.min_num_frame,
        max_sample_stride=args.max_sample_stride,
        min_sample_stride=args.min_sample_stride,
        train_ratio=args.train_ratio,
    )
    dataset.data_list = dataset.data_list * 100
    accelerator.print(f"Enlarged length of vkitti_video: {len(dataset)}")
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )


def get_data(data, args):
    # print(f"data {data if isinstance(data,str) else type(data)}")
    input_data = {
        "images": data["images"],
        "disparity": data["disparity"],
        # Extra images
        "extra_images": data.get("extra_images", None),
        "extra_image_frame_index": data.get("extra_image_frame_index", None),
        # Shape
        "batch_size": data["images"].shape[0],
        "num_frames": data["images"].shape[1],
        "height": data["images"].shape[-2],
        "width": data["images"].shape[-1],
    }
    return input_data


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x
    ):
        self.output_path = output_path
        import time
        os.makedirs(self.output_path, exist_ok=True)


def launch_training_task(
    accelerator,
    start_epoch,
    global_step,
    args,
    dataset_range,
    train_dataloader_list,
    test_loader_dict,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    validate_step: int = 500,
    log_step: int = 10,
):
    global _last_known_global_step
    validator = Validation()
    accelerator.print(
        f"Initial accelerator with gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    )
    accelerator.print(
        f"Using {accelerator.num_processes} processes for training.")

    accelerator.print(
        f"accelerator.state.deepspeed_plugin: {accelerator.state.deepspeed_plugin}")

    accelerator.print(f"Validate every {validate_step} steps.")

    loss_csv_path = os.path.join(model_logger.output_path, "loss_log.csv")
    if accelerator.is_main_process and not os.path.exists(loss_csv_path):
        with open(loss_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "depth_loss", "grad_loss", "learning_rate"])

    def _run_validation(current_global_step, save_training_state=False, reason="periodic"):
        model.pipe.dit.eval()
        if save_training_state:
            print(f"GPU {accelerator.process_index} saving training state...")
            accelerator.save_state(
                os.path.join(
                    model_logger.output_path, f"checkpoint-step-{current_global_step}"
                )
            )
            if accelerator.is_main_process:
                torch.save(
                    {"global_step": current_global_step},
                    os.path.join(model_logger.output_path, "trainer_state.pt"),
                )
                accelerator.print(
                    f"Checkpoint saved at step {current_global_step}"
                )

        gc.collect()
        _log_cuda_memory(
            accelerator,
            f"before {reason} validation step {current_global_step}",
        )
        _reset_cuda_peak_memory_stats(accelerator)
        validator.validate(
            accelerator=accelerator,
            dataset_range=dataset_range,
            pipe=model.pipe,
            global_step=current_global_step,
            args=args,
            test_loader_dict=test_loader_dict,
            output_path=model_logger.output_path,
        )
        gc.collect()
        _empty_cuda_cache(accelerator)
        _log_cuda_memory(
            accelerator,
            f"after {reason} validation step {current_global_step}",
        )

        model.pipe.scheduler.set_timesteps(
            training=True,
            denoise_step=args.denoise_step,
        )
        model.pipe.dit.train()
        _log_cuda_memory(
            accelerator,
            f"after {reason} validation reset step {current_global_step}",
        )

    if args.init_validate:
        accelerator.print(
            f"Starting validation with model at epoch {start_epoch}, global step {global_step}"
        )
        _run_validation(
            current_global_step=global_step,
            save_training_state=False,
            reason="initial",
        )
    accelerator.wait_for_everyone()

    optimizer.zero_grad()
    accumulate_depth_loss = 0.0
    accumulate_grad_loss = 0.0
    logged_batch_shapes = 0

    acm_cnt = 0
    rank = accelerator.process_index

    loader_iter_list = [iter(_train_dataloader)
                        for _train_dataloader in train_dataloader_list]
    prob = args.get('prob', [1 for _ in range(len(train_dataloader_list))])
    grad_loss = GradientLoss3DSeparate()

    if len(prob) != len(train_dataloader_list):
        raise ValueError(
            f"Expected {len(train_dataloader_list)} dataset probabilities, got {len(prob)}."
        )
    if sum(prob) <= 0:
        raise ValueError("At least one training dataset must have a positive probability.")
    set_seed(42)

    # Re-register signal handlers in case accelerate/torch overrode them.
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    print(f"{rank} Entering training loop...")
    _last_known_global_step = global_step

    for epoch_id in range(num_epochs):
        progress_bar = tqdm(
            range(100_000),
            desc=f"Epoch {epoch_id + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )
        progress_bar.set_postfix(global_step=global_step)
        for small_batch_step in progress_bar:
            should_run_validation = False
            input_data = None
            res_dict = None
            depth_gt = None
            pred = None
            pred_rgb = None
            pred_depth = None
            loss = None
            _grad_loss = None
            _grad_t = None
            _grad_h = None
            _grad_w = None
            select_pos = random.choices(
                population=range(len(train_dataloader_list)),
                weights=prob,
                k=1
            )[0]

            data = None
            try:
                data = next(loader_iter_list[select_pos])
            except StopIteration:
                print(
                    f"GPU used up dataset {select_pos}, setting up new one...")
                loader_iter_list[select_pos] = iter(
                    train_dataloader_list[select_pos])
                data = next(loader_iter_list[select_pos])

            # Forward and backward pass
            with accelerator.accumulate(model):
                input_data = get_data(data, args=args)
                if logged_batch_shapes < 4:
                    logged_batch_shapes += 1
                    _progress_log(
                        "Training batch shape "
                        f"batch={input_data['batch_size']} "
                        f"frames={input_data['num_frames']} "
                        f"height={input_data['height']} "
                        f"width={input_data['width']}"
                    )
                    _log_cuda_memory(
                        accelerator,
                        f"before forward batch {logged_batch_shapes}",
                    )
                try:
                    res_dict = model(input_data, args=args)
                except torch.OutOfMemoryError:
                    _progress_log(
                        "OOM during training forward "
                        f"batch={input_data['batch_size']} "
                        f"frames={input_data['num_frames']} "
                        f"height={input_data['height']} "
                        f"width={input_data['width']} "
                        "approx_wan_tokens="
                        f"{_wan_latent_tokens(input_data['num_frames'], input_data['height'], input_data['width'])}"
                    )
                    _log_cuda_memory(accelerator, "after training forward OOM")
                    raise
                depth_gt = res_dict['depth_gt']
                pred = res_dict['pred']

                # from torchvision.utils import save_image
                pred_rgb, pred_depth = None, None

                if isinstance(pred, tuple):
                    pred_depth, pred_rgb = pred
                else:
                    pred_depth = pred
                
                loss = torch.nn.functional.mse_loss(
                    depth_gt, pred_depth)
                
                
                accumulate_depth_loss += loss.item()

                if args.get('grad_loss', False):
                    _grad_loss = grad_loss(pred_depth, depth_gt)
                    _grad_t, _grad_h, _grad_w = _grad_loss
                    grad_co = args.get('grad_co', 1)
                    use_latent_flow = args.get('use_latent_flow', True)
                    if not use_latent_flow:
                        _grad_t = 0
                    loss += grad_co * (_grad_t+_grad_h+_grad_w)
                accumulate_grad_loss += loss.item()
                
                if accelerator.is_main_process:
                    _progress_log(
                        f"Small batch step {small_batch_step} total loss: {loss.item()} "
                    )
                accelerator.backward(loss)
                acm_cnt += 1

                # Update optimizer and scheduler
                if accelerator.sync_gradients:
                    
                    if args.get('clip_grad_norm', True):
                        accelerator.clip_grad_norm_(
                            model.trainable_modules(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    _last_known_global_step = global_step
                    if accelerator.is_main_process:
                        progress_bar.set_postfix(global_step=global_step)
                        _progress_log(
                            f"Global step at {global_step}, small batch step {small_batch_step}"
                        )

                    # Calculate the average loss across all processes
                    if global_step % log_step == 0:
                        accumulate_depth_loss /= acm_cnt
                        accumulate_grad_loss /= acm_cnt

                        accumulate_grad_loss = accumulate_grad_loss - accumulate_depth_loss

                        lr = scheduler.get_last_lr()[0]
                        _progress_log(
                            f"GPU {rank} step {global_step}: depth loss = "
                            f"{accumulate_depth_loss:.6f}, grad_loss = "
                            f"{accumulate_grad_loss:.6f}, learning rate : "
                            f"{lr:.8f}"
                        )
                        if accelerator.is_main_process:
                            with open(loss_csv_path, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([global_step, accumulate_depth_loss, accumulate_grad_loss, lr])
                        accumulate_depth_loss = 0.0
                        accumulate_grad_loss = 0.0
                        acm_cnt = 0

                    if (global_step) % validate_step == 0:
                        should_run_validation = True

            data = None
            input_data = None
            res_dict = None
            depth_gt = None
            pred = None
            pred_rgb = None
            pred_depth = None
            loss = None
            _grad_loss = None
            _grad_t = None
            _grad_h = None
            _grad_w = None

            if should_run_validation:
                _run_validation(
                    current_global_step=global_step,
                    save_training_state=True,
                    reason="periodic",
                )
            accelerator.wait_for_everyone()

            should_exit, global_step = maybe_save_and_exit_on_interrupt(
                accelerator=accelerator,
                model=model,
                model_logger=model_logger,
                global_step=global_step,
                args=args,
            )
            if should_exit:
                accelerator.end_training()
                return

        accelerator.end_training()


if __name__ == "__main__":
    def get_config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config", type=str, default=None, help="Path to config yaml"
        )
        args = parser.parse_args()
        cfg = OmegaConf.load(args.config)
        return cfg

    cfg = get_config()
    args = cfg

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[process_group_kwargs],
    )
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)
    accelerator.print(OmegaConf.to_yaml(cfg))

    # set_seed(42)

    # Save args
    os.makedirs(args.output_path, exist_ok=True)

    # Tee stdout/stderr to a log file so we don't need shell pipes.
    if accelerator.is_main_process:
        _log_path = os.path.join(
            args.output_path,
            f"train_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log",
        )
        _log_file = open(_log_path, "a")
        sys.stdout = _TeeStream(sys.__stdout__, _log_file)
        sys.stderr = _TeeStream(sys.__stderr__, _log_file)
        print(f"Logging to {_log_path}", flush=True)

    args_save_path = os.path.join(args.output_path, "args.yaml")

    if accelerator.is_main_process:
        sigint_decision_path = os.path.join(args.output_path, ".sigint_decision")
        if os.path.exists(sigint_decision_path):
            os.remove(sigint_decision_path)
        accelerator.print(f"Saving args to {args_save_path}")
        with open(args_save_path, "w") as f:
            f.write(OmegaConf.to_yaml(args))
    accelerator.wait_for_everyone()

    # Load model
    model = WanTrainingModule(
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        gradient_checkpoint_every_n=getattr(args, 'gradient_checkpoint_every_n', 1),
        lora_rank=args.lora_rank,
        lora_base_model=args.lora_base_model,
        args=args,
        accelerator=accelerator,
    )

    model.set_training_param()

    model_logger = ModelLogger(
        args.output_path,
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate)
    world_size = accelerator.num_processes

    if args.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.5, total_iters=args.warmup_steps*world_size
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    configured_prob = list(args.get("prob", []))
    if len(configured_prob) != 6:
        raise ValueError(
            "Expected `prob` to contain 6 entries in the order "
            "[hypersim_image, synthhuman_image, infinigen_image, vkitti_image, tartanair_video, vkitti_video]."
        )

    min_res = list(args.get("min_resolution", [])) or None
    max_res = list(args.get("max_resolution", [])) or None
    variable_resolution = min_res is not None and max_res is not None
    video_resolution_budget_num_frames = args.get(
        "video_resolution_budget_num_frames", 45
    )
    video_resolution_budget_scale = float(
        args.get("video_resolution_budget_scale", 1.0)
    )
    # When resolution varies per sample, image datasets must use batch_size=1
    # to avoid shape mismatches in the collate function.
    img_batch_size = 1 if variable_resolution else args.batch_size
    if variable_resolution:
        accelerator.print(
            f"Variable resolution enabled: min={min_res}, max={max_res}. "
            f"Image datasets will use batch_size=1."
        )
        accelerator.print(
            "Video frame/resolution pairs will be clipped to the "
            f"{args.resolution_hypersim[0]}x{args.resolution_hypersim[1]}x"
            f"{video_resolution_budget_num_frames} pixel-frame budget "
            f"with scale {video_resolution_budget_scale:.2f}."
        )

    dataset_builders = [
        (
            "hypersim_image",
            configured_prob[0],
            lambda: torch.utils.data.DataLoader(
                HypersimDataset(
                    data_dir=args.train_data_dir_hypersim,
                    resolution=args.resolution_hypersim,
                    random_flip=args.random_flip,
                    norm_type=args.norm_type,
                    truncnorm_min=args.truncnorm_min,
                    align_cam_normal=args.align_cam_normal,
                    split="train",
                    train_ratio=args.train_ratio,
                    min_resolution=min_res,
                    max_resolution=max_res,
                ),
                shuffle=True,
                batch_size=img_batch_size,
                num_workers=2,
                collate_fn=custom_collate_fn,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True,
                drop_last=True,
            ),
        ),
        (
            "synthhuman_image",
            configured_prob[1],
            lambda: torch.utils.data.DataLoader(
                SynthHumanDataset(
                    data_dir=args.train_data_dir_synthhuman,
                    resolution=args.resolution_hypersim,
                    random_flip=args.random_flip,
                    norm_type=args.norm_type,
                    truncnorm_min=args.truncnorm_min,
                    train_ratio=args.train_ratio,
                    min_resolution=min_res,
                    max_resolution=max_res,
                ),
                shuffle=True,
                batch_size=img_batch_size,
                num_workers=2,
                collate_fn=custom_collate_fn,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True,
                drop_last=True,
            ),
        ),
        (
            "infinigen_image",
            configured_prob[2],
            lambda: torch.utils.data.DataLoader(
                InfinigenDataset(
                    data_dir=args.train_data_dir_infinigen,
                    resolution=args.resolution_hypersim,
                    random_flip=args.random_flip,
                    norm_type=args.norm_type,
                    truncnorm_min=args.truncnorm_min,
                    train_ratio=args.train_ratio,
                    split_manifest=args.get("train_infinigen_manifest"),
                    min_resolution=min_res,
                    max_resolution=max_res,
                ),
                shuffle=True,
                batch_size=img_batch_size,
                num_workers=2,
                collate_fn=custom_collate_fn,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True,
                drop_last=True,
            ),
        ),
        (
            "vkitti_image",
            configured_prob[3],
            lambda: torch.utils.data.DataLoader(
                VKITTIDataset(
                    args.train_data_dir_vkitti,
                    norm_type=args.norm_type,
                    train_ratio=args.train_ratio,
                ),
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=2,
                collate_fn=custom_collate_fn,
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=True,
                drop_last=True,
            ),
        ),
        (
            "tartanair_video",
            configured_prob[4],
            lambda: _build_tartanair_video_dataloader(args, accelerator),
        ),
        (
            "vkitti_video",
            configured_prob[5],
            lambda: _build_vkitti_video_dataloader(args, accelerator),
        ),
    ]
    active_dataset_builders = [
        (name, prob, build_fn)
        for name, prob, build_fn in dataset_builders
        if prob > 0
    ]
    if not active_dataset_builders:
        raise ValueError("No active training datasets. Check the `prob` config.")

    active_dataset_names = [name for name, _, _ in active_dataset_builders]
    active_prob = [prob for _, prob, _ in active_dataset_builders]
    accelerator.print(
        f"Active training datasets: {active_dataset_names} with prob {active_prob}"
    )

    train_dataloader_list = [build_fn() for _, _, build_fn in active_dataset_builders]

    def _maybe_build_eval_loader(name, root_path, dataset_builder):
        if not root_path or str(root_path).startswith("your_"):
            accelerator.print(
                f"Skipping validation dataset `{name}`: root path is unset ({root_path})."
            )
            return None
        if not os.path.exists(root_path):
            accelerator.print(
                f"Skipping validation dataset `{name}`: path does not exist ({root_path})."
            )
            return None
        dataset = dataset_builder()
        accelerator.print(
            f"Enabled validation dataset `{name}` with {len(dataset)} samples from {root_path}."
        )
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.test_batch_size,
            num_workers=2,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )

    eval_loader_entries = []
    kitti_vid_test_dataloader = _maybe_build_eval_loader(
        "kitti",
        args.kitti_vid_test_data_root,
        lambda: KITTI_VID_Dataset(
            data_root=args.kitti_vid_test_data_root,
            max_num_frame=args.test_max_num_frame,
            min_num_frame=args.test_min_num_frame,
            max_sample_stride=args.test_max_sample_stride,
            min_sample_stride=args.test_min_sample_stride,
        ),
    )
    if kitti_vid_test_dataloader is not None:
        eval_loader_entries.append(("kitti", kitti_vid_test_dataloader))

    scannet_vid_test_dataloader = _maybe_build_eval_loader(
        "scannet",
        args.scannet_vid_test_data_root,
        lambda: Scannet_VID_Dataset(
            data_root=args.scannet_vid_test_data_root,
            split_ls=args.scannet_split_ls,
            test=False,
            max_num_frame=args.test_max_num_frame,
            min_num_frame=args.test_min_num_frame,
            max_sample_stride=args.test_max_sample_stride,
            min_sample_stride=args.test_min_sample_stride,
        ),
    )
    if scannet_vid_test_dataloader is not None:
        eval_loader_entries.append(("scannet", scannet_vid_test_dataloader))

    nyuv2_test_dataloader = _maybe_build_eval_loader(
        "nyuv2",
        args.nyuv2_test_data_root,
        lambda: NYUv2Dataset(
            data_root=args.nyuv2_test_data_root,
            test=False,
        ),
    )
    if nyuv2_test_dataloader is not None:
        eval_loader_entries.append(("nyuv2", nyuv2_test_dataloader))

    infinigen_eval_dataset_type = str(
        args.get("infinigen_test_dataset_type", "image")
    ).lower()
    if infinigen_eval_dataset_type == "video":
        infinigen_eval_builder = lambda: InfinigenVideoDataset(
            data_dir=args.infinigen_test_data_root,
            random_flip=False,
            norm_type=args.norm_type,
            truncnorm_min=args.truncnorm_min,
            resolution=args.resolution_hypersim,
            max_num_frame=args.test_max_num_frame,
            min_num_frame=args.test_min_num_frame,
            max_sample_stride=args.test_max_sample_stride,
            min_sample_stride=args.test_min_sample_stride,
            split_manifest=args.get("infinigen_test_manifest"),
            deterministic_sampling=True,
            resolution_budget_num_frames=video_resolution_budget_num_frames,
            resolution_budget_scale=video_resolution_budget_scale,
        )
    elif infinigen_eval_dataset_type == "image":
        infinigen_eval_builder = lambda: InfinigenDataset(
            data_dir=args.infinigen_test_data_root,
            random_flip=False,
            norm_type=args.norm_type,
            truncnorm_min=args.truncnorm_min,
            resolution=args.resolution_hypersim,
            split_manifest=args.get("infinigen_test_manifest"),
        )
    else:
        raise ValueError(
            "Expected `infinigen_test_dataset_type` to be either `image` or `video`."
        )

    infinigen_test_dataloader = _maybe_build_eval_loader(
        "infinigen",
        args.get("infinigen_test_data_root"),
        infinigen_eval_builder,
    )
    if infinigen_test_dataloader is not None:
        eval_loader_entries.append(("infinigen", infinigen_test_dataloader))

    start_epoch, global_step = 0, 0
    prepared = (
        accelerator.prepare(
            model,
            optimizer,
            scheduler,
            *train_dataloader_list,
            *[loader for _, loader in eval_loader_entries],
        )
    )
    model = prepared[0]
    optimizer = prepared[1]
    scheduler = prepared[2]
    _log_cuda_memory(accelerator, "after accelerator.prepare")
    num_train_dataloaders = len(train_dataloader_list)
    train_dataloader_list = list(prepared[3:3 + num_train_dataloaders])
    prepared_eval_loaders = prepared[3 + num_train_dataloaders:]

    if args.resume and args.training_state_dir is not None:

        _training_state_dir = args.training_state_dir
        if os.path.exists(_training_state_dir) and 'checkpoint' in _training_state_dir:

            accelerator.print(
                f"Resuming training state from {args.training_state_dir}...")
            # assign training state dir must come with a given global step
            global_step = args.global_step

        else:  # use the global step to find
            g_s_path = os.path.join(
                args.training_state_dir,  'trainer_state.pt')
            if os.path.exists(g_s_path):
                g_s_pt = torch.load(g_s_path)
                global_step = g_s_pt['global_step']

            args.training_state_dir = os.path.join(
                args.training_state_dir, 'checkpoint-step-{}'.format(global_step))

        if args.load_optimizer:
            accelerator.load_state(args.training_state_dir)
            for pg in optimizer.param_groups:
                pg['lr'] = args.learning_rate

        else:
            unwrapped_model = model
            while hasattr(unwrapped_model, "module"):
                unwrapped_model = unwrapped_model.module
            ckpt_path = os.path.join(
                args.training_state_dir, "model.safetensors")
            state_dict = load_file(ckpt_path, device="cpu")
            missing, unexpected = unwrapped_model.load_state_dict(
                state_dict, strict=False)
            assert len(unexpected) == 0

        if global_step > 0:
            accelerator.print(f"Resuming from global step {global_step}")
        else:
            accelerator.print(f"Training from scratch...")
        accelerator.print("Training state loaded.")
        accelerator.wait_for_everyone()

    if hasattr(model, 'module'):
        model = model.module

    test_loader_dict = {
        name: loader
        for (name, _), loader in zip(eval_loader_entries, prepared_eval_loaders)
    }

    dataset_range = {
        'kitti': [1e-5, 80],
        'scannet': [1e-3, 10],
        'nyuv2': [1e-3, 10],
        'infinigen': [1e-3, 65],
    }

    merged_args = OmegaConf.merge(args, {"prob": active_prob})
    try:
        launch_training_task(
            accelerator=accelerator,
            train_dataloader_list=train_dataloader_list,
            test_loader_dict=test_loader_dict,
            dataset_range=dataset_range,
            start_epoch=start_epoch,
            global_step=global_step,
            model=model,
            model_logger=model_logger,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            validate_step=args.validate_step,
            log_step=args.log_step,
            args=merged_args,
        )
    except KeyboardInterrupt:
        print(
            "\nKeyboardInterrupt received before the next safe point.",
            flush=True,
        )
        _save_checkpoint_on_interrupt(
            accelerator=accelerator,
            model=model,
            model_logger=model_logger,
            global_step=_last_known_global_step,
            args=merged_args,
        )
        accelerator.end_training()
