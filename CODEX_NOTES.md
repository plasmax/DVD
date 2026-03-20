## DVD Repo Notes

- Purpose: train and evaluate DVD, a deterministic video depth model built on Wan video diffusion backbones.
- Current training focus: a fresh single-GPU run on one 48 GB RTX A6000, not the original 8x H100 setup from the paper.
- Licensing constraint: do not train on KITTI video. In `examples/wanvideo/model_training/train_with_accelerate_video.py`, KITTI is only wired into evaluation/test loaders, not the training dataloader list, on both this branch and `main`.
- Training dataloader order in `train_with_accelerate_video.py`: `[hypersim_image, vkitti_image, tartanair_video, vkitti_video]`. The `prob` list in YAML must follow exactly that order.
- Effective sample exposure depends on both `prob` and loader batch size. Image loaders use `batch_size=args.batch_size`; video loaders are hard-coded to `batch_size=1`. If image `batch_size` is reduced, image `prob` usually needs to increase to preserve the original image/video balance.
- Original `main` config used `prob: [9, 1, 40, 10]` with `batch_size: 8`. That corresponds to approximate per-step sample exposure of `72, 8, 40, 10`.
- For the current A6000 run with `batch_size: 4` and VKITTI removed, `prob: [18, 0, 40, 0]` keeps image/video sample exposure closer to the original mix than `prob: [9, 0, 40, 0]`.
- `papers/DVD_2603.12250v1.md` and `papers/alphaxiv_summary.md` exist on this branch but not on `main`.
- If adding new datasets later, decide whether you want to preserve:
  - loader draw ratio: how often a loader is sampled;
  - sample exposure: how many actual training samples come from each loader after batch-size differences.
  For this repo, sample exposure is usually the more useful quantity.
- Partial-render / alpha-mask idea:
  - The current training path does not support alpha-conditioned inputs. Dataset images are loaded as RGB, and the model is trained on VAE latents from RGB plus disparity/depth targets.
  - The low-risk version is a `loss_mask` only: add a per-pixel mask in the dataset sample, stack it in the collate function, and use it only to weight the training loss.
  - In this repo the main depth loss is computed in latent space, not after VAE decode. A practical implementation would downsample the image-space mask to latent resolution and apply it as a float weighting tensor on the latent MSE (and, if desired, gradient loss).
  - The latent mask does not need to be boolean. A soft float mask is preferable because one latent cell corresponds to a patch of pixels; fractional values can represent partial coverage after downsampling.
  - This would let partially rendered CG frames contribute supervision only where valid depth exists, without changing the VAE input channel count or adding a new conditioning path.
