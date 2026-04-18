# Artifact Research Plan

Priority: Temporal Ghosting > Aspect Ratio Bars > Horizontal Lines > Haloing

## Research Phase (for the research agent -- internet access, no codebase/GPU)

(1) **Full spatio-temporal attention leakage.** Investigate how attention weight distributions shift as window size grows in video diffusion transformers. Do early-frame query tokens attend disproportionately to late-frame key tokens at larger window sizes? Research attention visualization and probing techniques applicable to a 30-block DiT with full spatio-temporal attention. Find any documented cases of temporal ghosting or information leakage in Wan 2.1, CogVideoX, HunyuanVideo, or similar full-attention video transformers.

(2) **3D RoPE extrapolation.** Research how rotary position embeddings behave when inference sequence lengths or spatial dimensions exceed the training distribution. Our training uses 21-61 frames but inference uses up to 81. Our training aspect ratios cluster around 4:3 but inference can be 16:9 or wider. Determine whether RoPE degradation is gradual or catastrophic, and whether known failure modes match our specific artifact signatures (ghosting for temporal extrapolation, vertical bars for spatial extrapolation). Survey solutions: NTK-aware scaling, YaRN, position interpolation, etc.

(3) **Window start vulnerability.** Our codebase already discards the first 16 frames of non-first windows because ghosting is worst there. Research why the temporal start of an attention window would be more susceptible to artifacts than the middle or end. Investigate whether this is a known property of causal attention warm-up, RoPE phase effects, or CausalConv3d padding behavior. Find any precedent in video generation or processing literature for "cold start" artifacts at window boundaries.

(4) **VAE temporal compression and boundary behavior.** Research the Wan 2.1 3D VAE architecture (from the paper and GitHub repo). How do CausalConv3d layers behave at sequence boundaries -- do they introduce first/last frame artifacts? How does the block causal mask in VAE attention affect information flow? What happens when the VAE decodes latents that are out of its training distribution (depth packed into 3-channel RGB format)? Document any known bf16 precision issues beyond nearest-neighbor interpolation.

(5) **Overlap, blending, and affine alignment.** Survey sliding-window strategies in video depth estimation (DepthCrafter, ChronoDepth, RollingDepth, DVD). Compare overlap/blending approaches: linear vs cosine blending, latent-space vs pixel-space alignment, affine vs non-linear alignment. Identify which strategies best suppress boundary artifacts and why.

(6) **LMR gradient loss and haloing.** Research how finite-difference gradient losses interact with sharp depth discontinuities in latent space. Find precedent for gradient-loss-induced haloing or edge overshoot in depth estimation or image restoration literature. Determine whether the effect is an inherent property of first-order finite differences or specific to the latent-space formulation.

(7) **Resolution-dependent artifacts and positional encoding aliasing.** Survey resolution-dependent artifacts in patch-based vision transformers. Can sinusoidal RoPE frequencies create visible interference patterns with regular spatial structures? Research resolution-invariant positional encoding approaches and any documented horizontal/vertical line artifacts in DiT-based systems.

(8) **Frozen generative VAE decoding non-native content.** The VAE was trained for RGB video but now decodes depth latents, with only channel 0 of 3 used as depth. Research whether decoding out-of-distribution latents through a frozen generative VAE introduces systematic artifacts. Find any precedent in Marigold, Lotus, DepthCrafter, or similar systems that repurpose generative VAEs for geometry prediction.

## Experimental Phase (for us to run locally -- requires codebase/GPU)

These experiments should be designed after the research phase, informed by findings above. Preliminary experiment list:

(A) **VAE-vs-DiT isolation.** Run the frozen VAE encode-decode roundtrip on ground-truth depth video without the DiT. Separately, feed ground-truth depth latents through the DiT and decode. Determines which component introduces each artifact.

(B) **Window-size sweep.** Inference on a fixed video with significant scene change at window sizes 21, 29, 37, 45, 53, 61, 69, 77, 81. Quantify ghosting intensity at each size to establish whether onset is gradual or threshold-based.

(C) **Attention weight visualization.** Using techniques identified in step (1), extract and visualize attention maps from the DiT during inference to detect temporal leakage patterns.

(D) **RoPE scaling experiments.** If step (2) identifies viable scaling methods (NTK-aware, YaRN, interpolation), test them at 81-frame inference to determine if ghosting is reduced.

(E) **Aspect ratio sweep.** Inference at aspect ratios from 1:1 to 2.39:1, keeping total pixel count constant, to map the relationship between aspect ratio and vertical bar width.

(F) **Trim-frames calibration.** Sweep trim_frames from 0 to 32 to find the minimum trim that eliminates ghosting at the window start, which reveals the temporal extent of the artifact.

(G) **LMR coefficient ablation.** Retrain with grad_co values of 0, 0.25, 0.5, 1.0, 2.0 and evaluate haloing intensity at depth boundaries.
