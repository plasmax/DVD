# Depth Dataset Follow-ups

## Status

- `SynthHuman` is currently wired as an image dataset, not a video dataset.
- `Infinigen` was added as an image dataset with a minimal loader and validator.

## Follow-ups

- Check `SynthHuman` sentinel background behavior.
  - The current training path converts EXR depth with `/ 100.0` and feeds it into the Hypersim-style normalization path.
  - The main `disparity` supervision path is not clamped before normalization.
  - Use [utils/visualize_synthhuman_depth.py](/net/code/workspaces/mlast/DVD/utils/visualize_synthhuman_depth.py) with `--report-stats` on representative files to see how much of the frame is at `65504` and whether that is materially affecting the low disparity percentiles.

- Decide whether `SynthHuman` background should stay as-is, be clamped, or be masked.
  - Current behavior: keep it in the normalization path.
  - Alternative options:
    - clamp far depth before normalization;
    - ignore invalid/background pixels with a loss mask;
    - replace sentinel values with a dataset-specific far-depth cap.

- Revisit the new `Infinigen` invalid-depth policy.
  - Current loader behavior fills non-finite depth with the farthest finite depth in-frame so training can proceed without masks.
  - This is pragmatic, not yet a deeply validated choice.
  - Check whether Infinigen depth should instead:
    - preserve `inf` until a masked loss is implemented;
    - use a fixed far-depth cap;
    - use a dataset-specific normalization path.

- Decide whether `Infinigen` should share Hypersim/SynthHuman preprocessing.
  - The current loader reuses `HypersimImageDepthNormalTransform`.
  - This is mechanically convenient, but not yet confirmed to be the right depth normalization policy for Infinigen.

- Partial-render / alpha-mask training is still only a design note.
  - The discussed low-risk path is a loss-only mask in latent space.
  - No training-time mask support has been implemented yet.
