# Training Progress Report for `saves/v005/loss_log.csv`

This note summarizes the current training run through global step `5750` using the statistics from `utils/analyse_log.py`.

## Executive Summary

Training appears healthy and stable.

- `depth_loss` has continued to improve through the end of the run.
- `grad_loss` improved substantially early on and now looks close to a plateau.
- Both losses are less noisy in the most recent part of training than in the preceding window.
- The run is now in a diminishing-returns regime rather than a failure or divergence regime.

At this point, the most likely next improvement is not "train much longer at the same settings" but "add validation and probably decay the learning rate once validation flattens."

## Current Run Snapshot

- Total logged points: `575`
- Step interval between logged points: `10`
- Latest global step: `5750`
- Learning rate: constant `1e-4`

Initial rows:

- Step `10`: `depth_loss=1.189233`, `grad_loss=0.517079`

Latest rows:

- Step `5750`: `depth_loss=0.027252`, `grad_loss=0.140645`

This means the run has already achieved a very large reduction in both objectives:

- `depth_loss` fell by roughly `97.7%` from the first logged point to the latest point.
- `grad_loss` fell by roughly `72.8%` from the first logged point to the latest point.

## What the Curves Suggest

### 1. Early optimization was strong

The first several hundred steps delivered the largest improvement. This is what we would hope to see when the optimizer first adapts the regression head and trainable components to the dataset.

The downsampled history shows the broad pattern clearly:

- Around step `310`: `depth_loss=0.086888`, `grad_loss=0.200760`
- Around step `1220`: `depth_loss=0.049097`, `grad_loss=0.160138`
- Around step `2430`: `depth_loss=0.036574`, `grad_loss=0.150752`

The run moved out of the "rapid descent" phase long ago and has been in fine-tuning territory for quite a while.

### 2. Late training is still productive, but only marginally

For `depth_loss`, the rolling averages at the end are the best seen so far:

- `last25=0.02762952`
- `last50=0.02799214`
- `last100=0.02852844`

The best rolling windows for `depth_loss` are all at the tail of the run:

- Best `25`-step window: steps `[5510, 5750]`
- Best `50`-step window: steps `[5260, 5750]`
- Best `100`-step window: steps `[4760, 5750]`

That is a strong sign that the model is still refining depth predictions rather than merely bouncing around.

For `grad_loss`, the story is slightly different:

- `last25=0.14113609`
- `last50=0.14062525`
- `last100=0.14160693`

The best `25`-step window for `grad_loss` happened a bit earlier:

- Best `25`-step window: steps `[5280, 5520]`

The best `50`- and `100`-step windows still include the late region, so `grad_loss` is not degrading in a serious way. It is simply no longer improving with the same consistency as `depth_loss`.

### 3. Noise is decreasing, not increasing

Late-stage volatility is lower than in the preceding 50-step window:

- `depth_loss`: recent/std ratio `0.741`
- `grad_loss`: recent/std ratio `0.853`

This is encouraging because it argues against unstable optimization. The model is settling down, which is what we want near convergence.

### 4. The two objectives are broadly aligned

Correlation between the two losses is `0.8507`, which is high.

That suggests improvements in one objective generally track improvements in the other. There is no obvious sign that one loss term is aggressively fighting the other at the current weighting.

## Best Regions Observed So Far

The best single-step results for both losses occurred at step `5020`:

- `depth_loss=0.02401709`
- `grad_loss=0.12027864`

That point is likely a locally favorable minibatch rather than the only checkpoint worth trusting. The rolling-window analysis is more informative than the best single step:

- The cleanest sustained `depth_loss` region is the end of the run.
- The cleanest sustained `grad_loss` region is roughly the `5280-5520` neighborhood.

This suggests that the most sensible checkpoints to keep for future validation are:

- Step `5020` for best single-step values
- A checkpoint around the center of the best `grad_loss` window
- A checkpoint near the latest step for best sustained `depth_loss`

## Plateau Estimate

Using the first point where a rolling-50 average comes within `2%` of the best rolling-50 value:

- `grad_loss` was already near-plateau by step `4520`
- `depth_loss` reached that same regime by step `5200`

This is a useful practical signal. The run is not "done" in the strict sense, but the easy gains are over.

## Interpretation

The most likely interpretation is:

- Optimization is working.
- The current setup is not diverging.
- The model is still learning fine depth detail.
- Structural-gradient supervision has mostly delivered its large gains already.
- Additional progress at fixed `1e-4` will probably be modest.

Without validation, we cannot yet tell whether the remaining training improvements correspond to better generalization or only better fit to the training distribution. Because of that, training loss alone is enough to say "the run is healthy," but not enough to say "the newest checkpoint is the best checkpoint."

## Recommended Next Steps

### Immediate priority

Add validation now. That is the missing piece needed to turn this from a healthy training run into a reliable model-selection process.

Useful validation outputs would include:

- scalar validation losses matching the training objectives
- at least one image-depth metric on held-out data
- qualitative saved predictions from fixed validation samples
- checkpoint-by-checkpoint comparison across the best candidate windows

### Training policy once validation exists

If validation is still improving:

- continue training
- consider reducing learning rate from `1e-4` to something like `3e-5` or `1e-5`

If validation is flat while training loss still improves slightly:

- treat the run as near convergence
- keep the best validation checkpoint rather than the latest checkpoint

If validation worsens while training loss improves:

- stop training
- use the best earlier checkpoint

## Bottom Line

This run is going well.

The model has already captured most of the large gains, remains stable, and still shows small but real improvement in `depth_loss`. The strongest current evidence points to a near-plateau regime rather than undertraining or instability. Validation is now the most important next addition because it will decide whether continued training and learning-rate decay are worthwhile.
