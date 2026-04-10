# Variable Resolution & Clip Length Training Strategy

Resuming from step 9400 with two new sources of training variance to
improve robustness at inference time.

## Variable clip length

- `min_num_frame: 21`, `max_num_frame: 61` (was fixed at 45)
- Sampled per clip; constraint `num_frames % 4 == 1` still applies
  (valid values: 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61)
- No architectural changes needed: RoPE dynamically slices to the
  actual frame count, and video datasets already use batch_size=1

## Variable resolution

- `min_resolution: [352, 480]`, `max_resolution: [576, 768]`
- ~0.73x to ~1.2x of the original [480, 640]
- Sampled per item in `__getitem__`, H and W each rounded to nearest 32
- Video clips now couple sampled resolution to sampled clip length so the
  total pixel-frame budget stays close to the original `480x640x45`
  training envelope. Short clips can still reach the max resolution; long
  clips are sampled from the smaller end of the range.
- Image datasets automatically switch to batch_size=1 when enabled
  (required because torch.stack in the collate function needs uniform shapes)
- No architectural changes needed: RoPE handles variable spatial dims,
  VAE encodes/decodes at whatever size it receives
- VKITTI datasets (both image and video) are unaffected (prob=0, hardcoded
  aspect ratio 352x1216)
- Eval datasets are unaffected (fixed resolution for reproducibility)

## Warmup

- `warmup_steps: 150` (was 0)
- LinearLR scheduler ramps from 0.5x to 1x learning rate
- Rationale: two simultaneous distribution shifts on a model that spent
  9400 steps on fixed resolution/frame-count data. A short warmup lets
  gradients stabilize before full-magnitude updates.

## Config diff

```yaml
# Clip length
max_num_frame: 61   # was 45
min_num_frame: 21   # was 45

# Resolution
min_resolution: [352, 480]   # new
max_resolution: [576, 768]   # new
video_resolution_budget_num_frames: 45   # keeps video T*H*W near the old budget

# Warmup
warmup_steps: 150   # was 0
```

## Backward compatibility

Removing `min_resolution` / `max_resolution` from the config (or setting
them to empty lists) falls back to the original fixed-resolution behavior.
`resolution_hypersim` is still used as the base/default resolution.
