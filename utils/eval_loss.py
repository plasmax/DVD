"""Evaluate training losses (depth + gradient) on a checkpoint.

Usage examples:

  # Evaluate on hypersim (image dataset, default):
  python utils/eval_loss.py \
      --ckpt saves/v005/checkpoint-step-750 \
      --config train_config/normal_config/video_config_new.yaml \
      --dataset hypersim --num_batches 20

  # Evaluate on tartanair (video dataset):
  python utils/eval_loss.py \
      --ckpt saves/v005/checkpoint-step-750 \
      --config train_config/normal_config/video_config_new.yaml \
      --dataset tartanair --num_batches 10

  # Compare two checkpoints:
  python utils/eval_loss.py \
      --ckpt path/to/their/checkpoint \
      --config train_config/normal_config/video_config_new.yaml \
      --dataset hypersim --num_batches 20
"""

import argparse
import os
import sys

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.wanvideo.model_training.training_loss import GradientLoss3DSeparate
from examples.wanvideo.model_training.WanTrainingModule import WanTrainingModule
from examples.wanvideo.model_training.train_with_accelerate_video import (
    custom_collate_fn,
    get_data,
)


def build_dataloader(args, dataset_name, batch_size):
    """Build a dataloader for the requested dataset."""
    if dataset_name == "hypersim":
        from examples.dataset import HypersimDataset

        dataset = HypersimDataset(
            data_dir=args.train_data_dir_hypersim,
            resolution=args.resolution_hypersim,
            random_flip=False,
            norm_type=args.norm_type,
            truncnorm_min=args.truncnorm_min,
            align_cam_normal=args.get("align_cam_normal", False),
            split="train",
            train_ratio=args.train_ratio,
        )
    elif dataset_name == "tartanair":
        from examples.dataset import TartanAir_VID_Dataset

        dataset = TartanAir_VID_Dataset(
            data_root=args.train_data_dir_ttr_vid,
            max_num_frame=args.max_num_frame,
            min_num_frame=args.min_num_frame,
            max_sample_stride=args.max_sample_stride,
            min_sample_stride=args.min_sample_stride,
            norm_type=args.norm_type,
            truncnorm_min=args.truncnorm_min,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'hypersim' or 'tartanair'.")

    return torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
    )


@torch.no_grad()
def evaluate(model, dataloader, num_batches, grad_co, use_latent_flow):
    """Run forward pass on data and compute training losses."""
    grad_loss_fn = GradientLoss3DSeparate()

    depth_losses = []
    grad_losses = []
    total_losses = []

    model.pipe.dit.eval()

    for i, data in enumerate(tqdm(dataloader, total=num_batches, desc="Evaluating")):
        if i >= num_batches:
            break

        input_data = get_data(data, args=None)
        res_dict = model(input_data)

        depth_gt = res_dict["depth_gt"]
        pred = res_dict["pred"]
        pred_depth = pred[0] if isinstance(pred, tuple) else pred

        # Depth MSE loss
        depth_loss = torch.nn.functional.mse_loss(depth_gt, pred_depth)
        depth_losses.append(depth_loss.item())

        # Gradient loss
        _grad_t, _grad_h, _grad_w = grad_loss_fn(pred_depth, depth_gt)
        if not use_latent_flow:
            _grad_t = 0
        gl = grad_co * (_grad_t + _grad_h + _grad_w)
        if isinstance(gl, torch.Tensor):
            gl = gl.item()
        grad_losses.append(gl)

        total_losses.append(depth_loss.item() + gl)

    return {
        "depth_loss": sum(depth_losses) / len(depth_losses),
        "grad_loss": sum(grad_losses) / len(grad_losses),
        "total_loss": sum(total_losses) / len(total_losses),
        "num_batches": len(depth_losses),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate training loss on a checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint dir (contains model.safetensors)")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--dataset", default="hypersim", choices=["hypersim", "tartanair"])
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches to evaluate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size (default: from config)")
    cli_args = parser.parse_args()

    args = OmegaConf.load(cli_args.config)
    accelerator = Accelerator()

    # Load model
    print(f"Loading model...")
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=args.lora_rank,
        lora_base_model=args.lora_base_model,
        args=args,
    )

    # Load checkpoint weights
    ckpt_path = os.path.join(cli_args.ckpt, "model.safetensors")
    print(f"Loading checkpoint from {ckpt_path}")
    dit_state_dict = {
        k.replace("pipe.dit.", ""): v
        for k, v in load_file(ckpt_path, device="cpu").items()
        if "pipe.dit." in k
    }
    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    model.merge_lora_layer()
    model = model.to("cuda")

    # Set scheduler for training-mode forward pass (matches how losses are computed)
    model.pipe.scheduler.set_timesteps(
        training=True,
        denoise_step=args.denoise_step,
    )

    # Build dataloader
    batch_size = cli_args.batch_size or args.batch_size
    print(f"Building {cli_args.dataset} dataloader (batch_size={batch_size})...")
    dataloader = build_dataloader(args, cli_args.dataset, batch_size)

    # Evaluate
    grad_co = args.get("grad_co", 1)
    use_latent_flow = args.get("use_latent_flow", True)
    results = evaluate(model, dataloader, cli_args.num_batches, grad_co, use_latent_flow)

    # Print results
    print("\n" + "=" * 50)
    print(f"Checkpoint:  {cli_args.ckpt}")
    print(f"Dataset:     {cli_args.dataset}")
    print(f"Batches:     {results['num_batches']}")
    print(f"{'=' * 50}")
    print(f"depth_loss:  {results['depth_loss']:.6f}")
    print(f"grad_loss:   {results['grad_loss']:.6f}")
    print(f"total_loss:  {results['total_loss']:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
