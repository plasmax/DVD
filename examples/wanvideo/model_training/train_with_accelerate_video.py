import argparse
import gc
import os
import random
import signal
import sys
import time
from datetime import timedelta
from itertools import cycle
from pathlib import Path

import numpy as np
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
from examples.dataset.hypersim_dataset import HypersimImageDepthNormalTransform
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


def sanitize_infinigen_depth(depth):
    finite_positive = np.isfinite(depth) & (depth > 0)
    if not finite_positive.any():
        raise ValueError("depth has no finite positive values")
    sanitized = depth.astype(np.float32, copy=True)
    # Infinigen commonly uses inf background; replace it with the farthest
    # finite depth so the existing normalization path can run without masks.
    sanitized[~finite_positive] = np.max(sanitized[finite_positive])
    return sanitized


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
            image = Image.open(img_path).convert("RGB")

            # SynthHuman depth is published in centimeters; convert to meters.
            depth = load_synthhuman_depth_exr(dep_path) / 100.0
            if not np.isfinite(depth).all():
                raise ValueError("depth contains NaN or inf")
            if (depth <= 0).any():
                raise ValueError("depth contains non-positive values")

            raw_depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            raw_depth = F.interpolate(
                raw_depth, size=(self.new_h, self.new_w), mode="nearest"
            ).squeeze()
            raw_depth = torch.clamp(raw_depth, 1e-3, 65).repeat(3, 1, 1)

            image, depth, normal = self.transform(image, depth, None)
            if torch.isnan(image).any() or torch.isinf(image).any():
                raise ValueError("image is nan or inf after transform")
            if torch.isnan(depth).any() or torch.isinf(depth).any():
                raise ValueError("depth is nan or inf after transform")

            return {
                "sample_idx": torch.tensor(idx),
                "images": image.unsqueeze(0),
                "disparity": depth.unsqueeze(0),
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
    ):
        self.data_dir = data_dir
        self.data_list = []
        self.invalid_indices = set()
        self.reported_invalid_indices = set()

        for image_path in sorted(Path(data_dir).rglob("Image*.png")):
            depth_name = image_path.name.replace("Image", "Depth", 1).rsplit(".", 1)[0] + ".npy"
            if "/Image/" in str(image_path):
                depth_path = Path(str(image_path).replace("/Image/", "/Depth/")).with_name(depth_name)
            else:
                depth_path = image_path.with_name(depth_name)
            if depth_path.exists():
                self.data_list.append((str(image_path), str(depth_path)))

        self.data_list = self.data_list[start:]
        if not self.data_list:
            raise RuntimeError(
                f"No Infinigen RGB/depth pairs found under {data_dir}"
            )

        new_h, new_w = resolution
        self.new_h = new_h
        self.new_w = new_w
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
            image = Image.open(img_path).convert("RGB")

            depth = np.load(dep_path)
            if depth.ndim != 2:
                raise ValueError(f"expected HxW depth, got shape {depth.shape}")
            depth = sanitize_infinigen_depth(depth)

            raw_depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            raw_depth = F.interpolate(
                raw_depth, size=(self.new_h, self.new_w), mode="nearest"
            ).squeeze()
            raw_depth = torch.clamp(raw_depth, 1e-3, 65).repeat(3, 1, 1)

            image, depth, normal = self.transform(image, depth, None)
            if torch.isnan(image).any() or torch.isinf(image).any():
                raise ValueError("image is nan or inf after transform")
            if torch.isnan(depth).any() or torch.isinf(depth).any():
                raise ValueError("depth is nan or inf after transform")

            return {
                "sample_idx": torch.tensor(idx),
                "images": image.unsqueeze(0),
                "disparity": depth.unsqueeze(0),
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
    dataset = TartanAir_VID_Dataset(
        data_dir=args.train_data_dir_ttr_vid,
        random_flip=args.random_flip,
        norm_type=args.norm_type,
        max_num_frame=args.max_num_frame,
        min_num_frame=args.min_num_frame,
        max_sample_stride=args.max_sample_stride,
        min_sample_stride=args.min_sample_stride,
        train_ratio=args.train_ratio,
    )
    dataset.data_list = dataset.data_list * 100
    accelerator.print(f"Enlarged length of tartanair_video: {len(dataset)}")
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

    if args.init_validate:
        accelerator.print(
            f"Starting validation with model at epoch {start_epoch}, global step {global_step}"
        )
        model.pipe.dit.eval()
        # Initial validation disabled for quick smoke runs.
        # validator.validate(
        #     accelerator=accelerator,
        #     dataset_range=dataset_range,
        #     pipe=model.pipe,
        #     global_step=global_step,
        #     args=args,
        #     test_loader_dict=test_loader_dict,
        #     output_path=model_logger.output_path,
        # )

        model.pipe.scheduler.set_timesteps(
            training=True,
            denoise_step=args.denoise_step,
        )
        model.pipe.dit.train()
    accelerator.wait_for_everyone()

    optimizer.zero_grad()
    accumulate_depth_loss = 0.0
    accumulate_grad_loss = 0.0

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
                res_dict = model(input_data, args=args)
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

                        _progress_log(
                            f"GPU {rank} step {global_step}: depth loss = "
                            f"{accumulate_depth_loss:.6f}, grad_loss = "
                            f"{accumulate_grad_loss:.6f}, learning rate : "
                            f"{scheduler.get_last_lr()[0]:.8f}"
                        )
                        accumulate_depth_loss = 0.0
                        accumulate_grad_loss = 0.0
                        acm_cnt = 0

                    if (global_step) % validate_step == 0:
                        model.pipe.dit.eval()
                        print(f"GPU {rank} saving training state...")
                        accelerator.save_state(
                            os.path.join(model_logger.output_path,
                                         f"checkpoint-step-{global_step}")
                        )
                        if accelerator.is_main_process:

                            torch.save(
                                {"global_step": global_step},
                                os.path.join(
                                    model_logger.output_path, "trainer_state.pt")
                            )
                            accelerator.print(
                                f"Checkpoint saved at step {global_step}")
                        # Validation disabled for quick smoke runs.
                        # validator.validate(
                        #     accelerator=accelerator,
                        #     pipe=model.pipe,
                        #     dataset_range=dataset_range,
                        #     global_step=global_step,
                        #     args=args,
                        #     test_loader_dict=test_loader_dict,
                        #     output_path=model_logger.output_path,
                        # )

                        model.pipe.scheduler.set_timesteps(
                            training=True,
                            denoise_step=args.denoise_step,
                        )
                        model.pipe.dit.train()
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
    # Test set
    # kitti_vid_test_dataset = KITTI_VID_Dataset(
    #     data_root=args.kitti_vid_test_data_root,
    #     max_num_frame=args.test_max_num_frame,
    #     min_num_frame=args.test_min_num_frame,
    #     max_sample_stride=args.test_max_sample_stride,
    #     min_sample_stride=args.test_min_sample_stride,
    # )
    # scannet_vid_test_dataset = Scannet_VID_Dataset(
    #     data_root=args.scannet_vid_test_data_root,
    #     split_ls=args.scannet_split_ls,
    #     test=False,
    #     max_num_frame=args.test_max_num_frame,
    #     min_num_frame=args.test_min_num_frame,
    #     max_sample_stride=args.test_max_sample_stride,
    #     min_sample_stride=args.test_min_sample_stride,
    # )
    # nyuv2_test_dataset = NYUv2Dataset(
    #     data_root=args.nyuv2_test_data_root,
    #     test=False,
    # )
    # kitti_vid_test_dataloader = torch.utils.data.DataLoader(
    #     kitti_vid_test_dataset,
    #     shuffle=False,
    #     batch_size=args.test_batch_size,
    #     num_workers=2,
    #     collate_fn=custom_collate_fn,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )
    # scannet_vid_test_dataloader = torch.utils.data.DataLoader(
    #     scannet_vid_test_dataset,
    #     shuffle=False,
    #     batch_size=args.test_batch_size,
    #     num_workers=2,
    #     collate_fn=custom_collate_fn,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )
    # nyuv2_test_dataloader = torch.utils.data.DataLoader(
    #     nyuv2_test_dataset,
    #     shuffle=False,
    #     batch_size=args.test_batch_size,
    #     num_workers=2,
    #     collate_fn=custom_collate_fn,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )
    kitti_vid_test_dataloader = []
    scannet_vid_test_dataloader = []
    nyuv2_test_dataloader = []

    start_epoch, global_step = 0, 0
    prepared = (
        accelerator.prepare(
            model,
            optimizer,
            scheduler,
            *train_dataloader_list,
            kitti_vid_test_dataloader,
            scannet_vid_test_dataloader,
            nyuv2_test_dataloader,
        )
    )
    model = prepared[0]
    optimizer = prepared[1]
    scheduler = prepared[2]
    train_dataloader_list = list(prepared[3:-3])
    kitti_vid_test_dataloader, scannet_vid_test_dataloader, nyuv2_test_dataloader = prepared[-3:]

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
            unwrapped_model = accelerator.unwrap_model(model)
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
        # 'kitti': kitti_vid_test_dataloader,
        # 'scannet': scannet_vid_test_dataloader,
        # 'nyuv2': nyuv2_test_dataloader
    }

    dataset_range = {
        'kitti': [1e-5, 80],
        'scannet': [1e-3, 10],
        'nyuv2': [1e-3, 10],
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
