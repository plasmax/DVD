# Research Prompt: Artifact Analysis in Deterministic Video Depth Estimation

## Context and Objective

You are a research agent tasked with producing a detailed technical report on the sources of visual artifacts in a **deterministic video depth estimation** system. The system is based on the **DVD** (Deterministic Video Depth) framework, which uses the **Wan 2.1 1.3B** video diffusion model as its backbone and performs single-pass depth regression (not iterative denoising). Your report will directly inform the design of controlled experiments to isolate and eliminate each artifact type.

You do **not** have access to the codebase, but you **do** have access to the internet. The two primary papers are:

- **DVD**: "Deterministic Video Depth Estimation with Generative Priors" (March 2026) -- https://github.com/EnVision-Research/DVD / https://dvd-project.github.io/
- **Wan 2.1**: "Wan: Open and Advanced Large-Scale Video Generative Models" (March 2025) -- https://github.com/Wan-Video/Wan2.1

---

## System Architecture Summary

The following describes the exact architecture and configuration in use. Read this section carefully -- every detail is relevant to the artifact analysis.

### Overall Pipeline

The system performs **single-pass deterministic depth regression** in VAE latent space:

```
Input RGB video  (B, 3, T, H, W)
       |
       v
  [Frozen VAE Encoder]  -->  RGB latents  (B, 16, T', H', W')
       |                         where H'=H/8, W'=W/8, T' ~ T/4
       v
  [DiT with LoRA]  -->  Depth latents  (B, 16, T', H', W')
       |                    single forward pass at fixed timestep tau=0.5
       v
  [Frozen VAE Decoder]  -->  Depth video  (B, 3, T, H, W)
       |
       v
  Take first channel  -->  Relative depth  (B, 1, T, H, W)
```

Key points:
- The VAE encoder and decoder are **frozen** (Wan 2.1's original 3D VAE, never fine-tuned)
- Only the DiT is trained, via **LoRA (rank 512)** on all attention projections (q, k, v, o) and FFN layers
- The model predicts depth in a **single forward pass** at a fixed timestep of 0.5 (not iterative denoising)
- The input to the DiT is the RGB latent (not noise), and the output is the predicted depth latent
- Depth is encoded as **truncated disparity** (1/depth), normalized to [-1, 1] using 2nd/98th percentile quantiles

### VAE Architecture (Wan 2.1's 3D VAE)

The VAE is critical to understanding several artifact types:

- **Encoder**: Uses `CausalConv3d` layers throughout -- these pad only on the left (past) side of the temporal dimension, enforcing temporal causality
- **Spatial compression**: 8x (three stages of 2x downsampling)
- **Temporal compression**: 4x (two stages with `temporal_downsample=[True, True, False]` across three blocks)
- **Latent channels**: 16 (the `z_dim` parameter is 4 in the config, but the actual latent has 16 channels)
- **Dimension multipliers**: [1, 2, 4, 4] across the four downsampling blocks
- **Attention**: Single-head self-attention blocks on the spatio-temporal grid inside the VAE bottleneck
- **Normalization**: RMS normalization (channel-first)
- **Upsampling fix**: The `Upsample` class casts to float32 for nearest-neighbor interpolation, then casts back -- this is specifically to work around **bfloat16** interpolation issues
- **Temporal cache**: `CACHE_T = 2` -- the VAE caches 2 temporal frames during sequential decoding
- **Block causal masking**: Attention inside the VAE uses block causal masks

The VAE's temporal compression from T to ~T/4 means that each latent frame effectively encodes information from ~4 input frames. The causal convolutions mean information flows forward in time but not backward.

### DiT Architecture (WanModel)

The Diffusion Transformer processes the patchified latent sequence:

- **Patch size**: [1, 2, 2] (temporal, height, width) -- no temporal patchification
- **Hidden dim**: 1536
- **Attention heads**: 12
- **Transformer blocks**: 30
- **FFN dim**: 8960
- **Input/output channels**: 16

**Attention mechanism**:
- **Full spatio-temporal self-attention**: Every token attends to every other token across all spatial and temporal positions within the window. There is no separation of spatial and temporal attention -- it is a single unified attention over the flattened (F, H, W) grid
- **Cross-attention**: With text embeddings (4096-dim), though in depth mode the text prompt is empty (zeros)
- **RoPE (3D Rotary Position Embeddings)**: Separate rotary embeddings for temporal (f), height (h), and width (w) dimensions. The dimension allocation is: `dim - 2*(dim//3)` for time, `dim//3` each for height and width. Precomputed up to sequence length 1024

**Key implication**: The full spatio-temporal attention means every spatial position at every temporal position within a window can attend to every other. The RoPE encodes 3D position (frame, row, column). This is fundamentally different from architectures that separate spatial and temporal attention.

### Training Configuration

- **Training resolution**: Variable, sampled from a grid between min=[352, 480] and max=[576, 768], aligned to 32-pixel boundaries
- **Training window size**: Variable, between 21 and 61 frames (config: `min_num_frame: 21`, `max_num_frame: 61`)
- **Resolution budget**: Constrained so that `H * W * num_frames * 0.9` stays under a budget calibrated at 45 frames
- **Aspect ratios**: Multiple aspect ratios are sampled during training (all combinations within the min/max resolution grid)
- **Timestep**: Fixed at `denoise_step: 0.5` (tau = 500/1000)
- **Training target**: `"x"` (predict the depth latent directly, not noise)
- **LoRA rank**: 512, targeting modules `["q", "k", "v", "o", "ffn.0", "ffn.2"]`
- **Loss**: L2 depth prediction loss + 0.5 * LMR gradient loss (temporal + height + width components)
- **Precision**: bf16 mixed precision
- **Datasets**: Hypersim (images), TartanAir (video), Virtual KITTI (images + video), SynthHuman (images) -- 367K total frames, all synthetic
- **Image-video joint training**: Batches contain both static images (F=1) and video sequences

### LMR (Latent Manifold Rectification) Loss

The LMR loss computes first-order finite differences in three directions and penalizes the difference between predicted and target gradients:

```
L_temporal = |pred[:,:,1:,:,:] - pred[:,:,:-1,:,:] - (target[:,:,1:,:,:] - target[:,:,:-1,:,:])|
L_height   = |pred[:,:,:,1:,:] - pred[:,:,:,:-1,:] - (target[:,:,:,1:,:] - target[:,:,:,:-1,:])|
L_width    = |pred[:,:,:,:,1:] - pred[:,:,:,:,:-1] - (target[:,:,:,:,1:] - target[:,:,:,:,:-1])|
```

These operate in **latent space** (on the 16-channel latent), not in pixel space.

### Inference Configuration

Default inference settings (from the original DVD repository):
- **Resolution**: 480x640 (height x width)
- **Window size**: 81 frames
- **Overlap**: 21 frames (stride = 60)

Our modified inference settings:
- **Resolution**: Variable (we test different resolutions)
- **Window size**: Variable (we test 45, 61, 81, and other sizes)
- **Overlap**: Variable
- **Trim frames**: 16 (the first 16 frames of each non-first window are discarded as "warmup" and re-inferred in the overlap region)

**Temporal padding**: Input frame counts must satisfy `T % 4 == 1` (matching the VAE's 4x temporal compression). If not, the last frame is repeated to pad.

**Sliding window strategy (original DVD)**:
- Windows are processed independently through the full pipeline (VAE encode -> DiT -> VAE decode)
- Overlapping regions are aligned using **least-squares affine alignment** (scale and shift computed from the overlap)
- Blending uses **linear interpolation** across the overlap region
- The scale is clamped to [0.7, 1.5] for stability

**Alternative sliding window strategy (in-pipeline temporal tiling)**:
- The pipeline also supports tiling in latent space before VAE decoding
- Uses a `TemporalTiler_BCTHW` class that applies **linear fade masks** at window boundaries
- Accumulates weighted predictions and normalizes

---

## Artifact Types Under Investigation

We observe four distinct artifact types during inference. For each, we provide a detailed description, the conditions under which it manifests, and our current hypotheses. **Your task is to research the likely root causes of each and provide evidence-based analysis.**

### Artifact 1: Aspect Ratio Bars

**Description**: When inferencing video with wider aspect ratios (e.g., 16:9 or wider), two fairly consistent vertical bars appear on the left and right edges of the depth output. These bars have a different depth character than the rest of the image -- they appear as a boundary region where the depth estimation degrades.

**Conditions**:
- Appears primarily at aspect ratios wider than what was used in training
- The bar width appears roughly consistent regardless of the input content
- More visible at wider aspect ratios (e.g., 2.39:1 cinema)

**Current hypothesis**: The training resolution grid spans min=[352, 480] to max=[576, 768], giving aspect ratios roughly between 1.33:1 (4:3) and 1.33:1 (all combinations favor roughly this ratio range). When inference is run at 16:9 or wider, the model encounters spatial positions it was never trained on. The RoPE position embeddings may not extrapolate well beyond the trained grid. The VAE may also behave differently at untrained aspect ratios.

**Interim mitigation**: Squash input to 4:3 aspect ratio before inference, then stretch the depth back.

### Artifact 2: Horizontal Lines (Interference Pattern)

**Description**: Faint horizontal lines appear across the depth output, predominantly in the lower half of the image, sometimes with a prominent stripe across the middle. Occasionally vertical lines appear too. The pattern has been described as looking like an "interference pattern" and also resembles bfloat16 quantization artifacts.

**Conditions**:
- Occurs most prominently at fixed resolutions (originally 640x480)
- Has been substantially reduced after switching to variable resolution training
- Still occasionally appears, though much less frequently
- May correlate with specific spatial frequencies in the input

**Current hypothesis**: This may be caused by multiple overlapping factors:
1. **Fixed resolution training**: When all training used 640x480, the model may have learned resolution-specific patterns in the positional embeddings
2. **bfloat16 precision**: The VAE's `Upsample` class explicitly works around bf16 nearest-neighbor interpolation issues (`super().forward(x.float()).type_as(x)`). Other bf16 operations in the pipeline may introduce similar quantization artifacts
3. **RoPE frequency aliasing**: At certain resolutions, the RoPE frequencies may alias with spatial features, creating interference patterns
4. **VAE decoder artifacts**: The 8x spatial upsampling through transposed convolutions may introduce checkerboard-like patterns at certain resolutions

### Artifact 3: Temporal Ghosting

**Description**: This is the most significant artifact. Semi-transparent "ghost" images of scene content appear overlaid on the depth output. The ghosting has these specific characteristics:
- Ghosts appear **primarily towards the beginning** of each inference window
- The ghosted content tends to reflect details that are present **towards the end** of the same window
- The effect is as though some detail between the start and end of the window is being "shared" -- fragments of late-window content appear at early-window positions
- The ghosting is **strongly correlated with window size**: much less at 45 frames, frequently present at 81 frames
- Ghosting is most visible in windows with **significant scene change** (e.g., camera panning across very different content)
- The effect is in the depth channel specifically, not an RGB passthrough artifact

**Conditions**:
- Worse at larger window sizes (81 frames >> 45 frames)
- Worse when the scene changes significantly within the window
- Present even with variable window sizes during training
- Switching to variable window sizes during training reduced it somewhat but did not eliminate it

**Current hypotheses (uncertain)**:
1. **Full spatio-temporal attention leakage**: Because the DiT uses full attention over all (F, H, W) tokens, information from late frames can directly influence early-frame token representations. If the attention patterns learned during training don't fully enforce temporal locality, end-of-window information could "leak" to the beginning
2. **VAE temporal compression**: The 4x temporal compression means each latent frame encodes ~4 input frames. The causal convolutions should prevent backward information flow, but the **attention blocks inside the VAE** use block causal masks that may allow some information sharing
3. **RoPE extrapolation at longer windows**: If the model was trained primarily on shorter windows (21-61 frames) but inferred at 81 frames, the RoPE embeddings at positions beyond the training distribution may cause attention weights to distribute incorrectly, allowing distant temporal positions to receive undue weight
4. **Latent space entanglement**: The depth and temporal information may be entangled in the 16-channel latent in ways that cause the VAE decoder to mix temporal information

**This artifact is the highest priority for investigation.**

### Artifact 4: Haloing (Dark Edge Separation)

**Description**: A soft dark halo appears at the boundaries between objects at different depths -- similar to a drop shadow effect. The halo creates a thin dark border separating foreground from background objects.

**Conditions**:
- Present across most inference settings
- More visible at sharp depth discontinuities
- Not hugely noticeable in 2D viewing but suggests geometry errors at depth boundaries
- Lower priority than the other artifacts

**Current hypothesis**: The LMR gradient loss penalizes differences in latent-space gradients between prediction and target. At sharp depth boundaries, the first-order finite difference approximation of the gradient creates a one-pixel-wide transition zone. The L2 depth loss encourages blending at boundaries (regression to the mean), while the LMR loss encourages sharpness. This tension may produce a "compromise" that manifests as a dark halo -- the model slightly overshoots the gradient correction, creating a boundary region that is darker than either the foreground or background depth values.

---

## Research Questions

For each artifact type, please research and report on the following:

### Section 1: VAE-Related Artifact Sources

1. **Wan 2.1 VAE architecture**: What is the exact encoder/decoder architecture of the Wan 2.1 3D VAE? How does the temporal compression work? What are the known limitations or failure modes?
2. **CausalConv3d behavior at boundaries**: How does causal temporal padding affect the first and last frames of a sequence? Are there known boundary artifacts in causal 3D convolution architectures?
3. **VAE temporal compression and information mixing**: With 4x temporal compression, each latent frame represents ~4 input frames. How is this temporal information distributed in the latent? Can the attention blocks inside the VAE cause information leakage between temporally distant frames?
4. **VAE decoder upsampling artifacts**: Do the transposed convolutions or nearest-neighbor upsampling in the decoder create systematic spatial artifacts? At what resolutions are these most likely?
5. **bfloat16 precision in the VAE**: What specific operations in the VAE are most sensitive to bf16 quantization? The codebase already fixes one (nearest-neighbor interpolation). Are there others?
6. **VAE behavior at untrained resolutions/aspect ratios**: If the VAE was trained on certain resolution ranges, how does it behave when processing inputs outside that range? Does the convolution padding or attention change behavior?

### Section 2: DiT Attention and Temporal Artifacts

7. **Full spatio-temporal attention in Wan 2.1**: How does full attention over all (F, H, W) tokens behave as the sequence length grows? Is there evidence in the literature of attention-based temporal leakage in video transformers?
8. **RoPE extrapolation behavior**: How do 3D rotary position embeddings behave when the inference sequence length exceeds training lengths? What are the known failure modes? How does the frequency allocation (time vs. spatial) affect this?
9. **Attention weight distribution at different window sizes**: In a deterministic depth regression setting (not generative), how should we expect attention to distribute across temporal positions? Is there theoretical or empirical evidence that attention patterns become less local at larger window sizes?
10. **Ghosting in video diffusion models**: Is temporal ghosting a known phenomenon in video diffusion transformers? What are the documented causes in the generative setting, and how might these differ in the deterministic regression setting?
11. **Window boundary effects in sliding-window video transformers**: What are the known artifacts at window boundaries in sliding-window video processing? How do different overlap and blending strategies affect these?

### Section 3: Training and Loss-Related Artifacts

12. **Regression to the mean in latent space**: How does the "mean collapse" phenomenon documented in the DVD paper manifest visually? Beyond what LMR addresses, are there known residual effects of deterministic regression in latent space?
13. **LMR gradient loss edge effects**: How does a finite-difference gradient loss interact with sharp depth boundaries in latent space? Can it create systematic over/undershoot at edges?
14. **Joint image-video training artifacts**: Are there known artifacts from mixing static images (F=1) and video sequences in training? Could this contribute to temporal inconsistencies?
15. **Synthetic-only training data**: All training data is synthetic (Hypersim, TartanAir, Virtual KITTI, SynthHuman). What are the known domain gap issues when training on synthetic depth and deploying on real-world video? Are there systematic bias patterns?
16. **LoRA rank and capacity**: At rank 512, the LoRA adapters are relatively high-rank. Could capacity limitations in the adaptation contribute to artifacts, particularly at the boundaries of what was trained?

### Section 4: Resolution and Positional Encoding

17. **Resolution-dependent artifacts in vision transformers**: What is the literature on resolution-dependent artifacts in ViTs? How do patch-based models handle resolution changes between training and inference?
18. **RoPE frequency design for 3D**: How are the rotary frequencies designed for the 3D case (temporal + 2D spatial)? What happens when the aspect ratio at inference differs from training? Are there known solutions for resolution-invariant positional encoding?
19. **Spatial frequency interference with positional encodings**: Can the sinusoidal positional encoding frequencies create visible interference patterns with regular spatial structures in the input (e.g., tiled floors, striped patterns)?

### Section 5: Experimental Design Recommendations

20. **Controlled experiments to isolate VAE vs DiT contributions**: What experimental setups would cleanly separate VAE-introduced artifacts from DiT-introduced artifacts? For example, encoding-decoding without the DiT, or running the DiT with ground-truth latents.
21. **Ablation strategies for temporal ghosting**: Given the architecture, what is the most efficient set of ablations to identify whether ghosting originates in (a) the VAE encoder, (b) the DiT attention, (c) the DiT positional encoding, or (d) the VAE decoder?
22. **Measuring attention patterns**: What tools or techniques exist for visualizing and quantifying attention weight distributions in large video transformers? How can we detect "attention leakage" between temporally distant tokens?
23. **Resolution sweep methodology**: How should we design a controlled resolution sweep to identify resolution-dependent artifacts while controlling for other variables (window size, content)?

---

## Output Format

Structure your report as follows:

1. **Executive Summary**: Key findings and highest-confidence hypotheses (1-2 pages)
2. **Per-Artifact Deep Dive**: For each of the four artifact types, provide:
   - Literature review of similar artifacts in related systems
   - Analysis of the most likely root cause(s) in our specific architecture
   - Confidence level (high/medium/low) for each hypothesis
   - Supporting evidence from papers, codebases, or known issues
3. **VAE Analysis**: Detailed analysis of the Wan 2.1 VAE's temporal behavior and known failure modes
4. **Attention and Positional Encoding Analysis**: How full spatio-temporal attention + 3D RoPE behave at the boundary conditions relevant to our artifacts
5. **Experimental Design**: Prioritized list of experiments, each with:
   - Hypothesis being tested
   - Independent/dependent variables
   - Expected outcome if hypothesis is correct vs. incorrect
   - Implementation complexity estimate (low/medium/high)
6. **References**: All papers, GitHub issues, and technical discussions cited

---

## Important Notes for Research

- The system is NOT a generative diffusion model at inference time. It performs a **single forward pass** at a fixed timestep. Many diffusion-model artifacts (sampling noise, mode mixing, CFG artifacts) do not apply here.
- The VAE is **frozen** -- it was never fine-tuned for depth. It was trained for video generation. This means the latent space is optimized for RGB video, not depth maps.
- The DiT was pre-trained for **video generation** and adapted via LoRA for depth regression. Its attention patterns and positional encodings were learned in the generative context.
- The depth output is derived from a **3-channel** output (the VAE decodes to 3 channels), from which only the **first channel** is used as depth. The other two channels are discarded. This is important -- the VAE is reconstructing a "video" where the depth information is packed into a format that wasn't part of the VAE's original training distribution.
- When researching the Wan 2.1 codebase (https://github.com/Wan-Video/Wan2.1), focus on the **1.3B model** architecture and the VAE implementation, as these are what DVD uses.
- When researching the DVD codebase (https://github.com/EnVision-Research/DVD), note that our local version has modifications including variable resolution training, variable window sizes, a trim-frames warmup strategy, and EXR sequence output support.
