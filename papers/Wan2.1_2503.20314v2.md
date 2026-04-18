# Wan: Open and Advanced Large-Scale Video Generative Models

## Authors
Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, Ziyu Liu

## Abstract
This report presents Wan, a comprehensive and open suite of video foundation models designed to push the boundaries of video generation. Built upon the mainstream diffusion transformer paradigm, Wan achieves significant advancements in generative capabilities through a series of innovations, including our novel VAE, scalable pre-training strategies, large-scale data curation, and automated evaluation metrics. These contributions collectively enhance the model's performance and versatility. Specifically, Wan is characterized by four key features: Leading Performance: The 14B model of Wan, trained on a vast dataset comprising billions of images and videos, demonstrates the scaling laws of video generation with respect to both data and model size. It consistently outperforms the existing open-source models as well as state-of-the-art commercial solutions across multiple internal and external benchmarks, demonstrating a clear and significant performance superiority. Comprehensiveness: Wan offers two capable models, i.e., 1.3B and 14B parameters, for efficiency and effectiveness respectively. It also covers multiple downstream applications, including image-to-video, instruction-guided video editing, and personal video generation, encompassing up to eight tasks. Consumer-Grade Efficiency: The 1.3B model demonstrates exceptional resource efficiency, requiring only 8.19 GB VRAM, making it compatible with a wide range of consumer-grade GPUs. Openness: We open-source the entire series of Wan, including source code and all models, with the goal of fostering the growth of the video generation community. This openness seeks to significantly expand the creative possibilities of video production in the industry and provide academia with high-quality video foundation models. All the code and models are available at https://github.com/Wan-Video/Wan2.1.



0.88



# Introduction

Since the introduction of Sora  by OpenAI, video generation technology
has attracted substantial attention from both industry and academia,
leading to rapid advancements in the field. The emergence of models
capable of generating videos that are on par with professionally crafted
content has significantly improved the efficiency of content creation
while simultaneously reducing the costs associated with video
production. These rapid advancements in video generation technology have
also been greatly attributed to the development of the open-source
community. Notable projects like HunyuanVideo , Mochi , and CogVideoX 
have made their video foundation model codes and weights publicly
available, gradually narrowing the gap between open-source models and
commercial counterparts. However, it is essential to acknowledge the
persistent gap between these outstanding open-source models and the
latest closed-source models. This gap is primarily evident in three
aspects: *Suboptimal Performance*: A notable performance gap remains, as
the pace of development in commercial models far exceeds that of current
open-source models, resulting in significantly superior capabilities.
*Limited Capabilities*: Most foundational models are limited to general
text-to-video (T2V) tasks, whereas the demands of video creation are
multifaceted. Consequently, basic T2V models are insufficient to address
these diverse requirements. *Insufficient Efficiency*: Despite their
impressive performance and scale, these models often prove impractical
for creative teams with limited computational resources, hindering their
accessibility and usability. These challenges collectively impose
constraints on the continued growth and innovation within the
open-source community.

To address the aforementioned challenges, this report introduces and
publicly releases a novel series of high-performance foundational video
generation models, referred to as , which sets a new benchmark in the
field. The core design of is inspired by the success of Diffusion
Transformers (DiT)  combined with Flow Matching , a framework that has
demonstrated the significant performance gains achievable through
scaling in both text-to-image (T2I)  and text-to-video (T2V)  tasks.
Within this architectural paradigm, cross-attention is employed to embed
text conditions, while the model’s design is meticulously optimized to
ensure computational efficiency and precise text controllability. To
further enhance the model’s ability to capture complex dynamics, a full
spatio-temporal attention mechanism is incorporated. Through extensive
experimentation, the model is validated at scale, reaching 14 billion
parameters. Subsequently, has seen large-scale data comprising billions
of images and videos, amounting to $\mathcal{O}(1)$ trillions of tokens
in total. This extensive training facilitates the emergence of the
model’s capabilities, allowing it to demonstrate robust performance
across multiple dimensions, such as motion amplitude and quality, visual
text generation, camera control, instruction adherence, and stylistic
diversity. Building upon the powerful foundational model, we have
expanded its capabilities to numerous downstream tasks, including
image-to-video generation (I2V), instruction-guided video editing (V2V),
zero-shot personalized customization, real-time video generation, and
audio generation, among other critical applications. To minimize
inference costs, we also introduce a 1.3B model alongside a 14B model
for T2V and I2V, both of which support 480p resolution and greatly
enhance inference efficiency. Remarkably, the 1.3B model requires only
8.19G of VRAM, allowing it to run on many consumer-grade GPUs, while its
performance exceeds that of many larger open-source models.

Moreover, we will also publicly present the entire training process,
including the large-scale data construction pipeline, video variational
autoencoder (VAE), training strategies, acceleration techniques, and
automated evaluation algorithms, to empower the community in developing
specialized foundational video models. In addition, we will provide
comprehensive design details and experimental results, offering insights
into phenomena observed during the resource-intensive training of large
generative models alongside our key findings and conclusions. We are
confident that these contributions will play a pivotal role in
accelerating the advancement of video generation technology.



# Related Work

Driven by advances in generative modeling, the landscape of large-scale
video models has evolved significantly, particularly in diffusion-based
frameworks. Our review focuses on two broad categories: closed-source
models and contributions from the open-source community.

**Closed-source models.** Closed-source models are mainly those
developed by major technology companies, which aims at high-quality,
professional video generation due to the extensive resources invested.
We organize the notable models released over the past year in
chronological order. In February 2024, OpenAI introduced Sora  and
marked a significant leap in AI-generated content. In June 2024, Kling 
by Kuaishou and Luma 1.0  by LUMA AI were opened up for public testing,
offering powerful video generation capabilities. Meanwhile, Runway
unveiled Gen-3 , building upon Gen-2 , to further elevate the standards
in video creation. In July 2024, Shengshu AI released Vidu  equipped
with self-designed U-ViT  architecture. In September 2024, both Kling
and Luma undergone updates to version 1.5. During the same period,
MiniMax introduced Hailuo Video , delivering impressive visual results
to the public. In October 2024, Pika Labs launched Pika 1.5 , enabling
users to customize visual and physical attributes in videos. In
addition, Meta introduced Movie Gen , a series of video foundation
models that detailed their training processes and applications. In
December 2024, Kling advanced to version 1.6 and, concurrently, Google
released Veo 2  with an improved understanding of real-world physics and
human movement nuances.

These developments underscore the intense global competition in the
video generation sector. In this context, our open-source has shown
competitive or even superior performance over these commercial models
across both internal and external benchmarks, leading in multiple
attributes.

**Contributions from open-source community.** On the other hand, the
open-source community has made substantial contributions not only to the
holistic video generative models but also to the exploration of
essential model components. Diffusion-based video generation models
typically build on Stable Diffusion  architecture. It mainly involves
three critical modules: an autoencoder to map the original video into a
compact latent space, a text encoder to extract text embeddings, and a
neural network optimized via the diffusion model to learn the
distribution of these video latents. For network structure, U-Net ,
initially used in image generation , is adapted for video generation by
incorporating temporal dimensions. VDM  extends the 2D U-Net to a 3D
version, while another paradigm  introduces 1D temporal attention
combined with 2D spatial attention block to reduce computation costs.
Notably, Diffusion Transformers (DiT ), which rely solely on transformer
blocks, are superior to U-Net in visual generation tasks . This
structure has also been transferred to video models , giving rise to two
common variants: the original DiT  that uses cross-attention for text
embeddings and MM-DiT , where text embeddings are concatenated to visual
embeddings for full-attention processing. Regarding autoencoders, while
the early approach  adopts standard VAE , recent autoencoders like
VQ-VAE  and VQGAN  improve model design for better reconstruction and
compression. LTX-Video  modifies the VAE decoder to perform the final
denoising step and convert latents into pixels, where the missing
high-frequency details are generated during decoding. The text encoder
is also critical for text-based video generation. Current powerful video
generation models primarily utilize the T5 series  as the main text
encoder, often in conjunction with CLIP . In the case of HunyuanVideo ,
T5 is replaced with a Multimodal Large Language Model  to achieve more
robust alignment between text embeddings and visual features.

By integrating these key modules with effective diffusion-based
optimization techniques , multiple promising open-source video
generation models have emerged . In , we have carefully designed or
selected each critical component to ensure high-quality video synthesis.
We provide detailed design information and ablation study, which can
facilitate the design of future video generation models.

In addition, numerous studies have explored downstream tasks in video
generation. These tasks include repainting , editing , controllable
generation , and frame reference generation , often by leveraging
adapter-based and ControlNet-like architectures  to incorporate
user-specified conditions. We also develop various downstream
applications based on , which have demonstrated excellent performance.

# Data Processing Pipeline

High-quality data is essential for training large generative models, and
an automated data construction pipeline significantly enhances the
efficiency of the training process. In developing our dataset, we
prioritized three core principles: *high quality*, *high diversity*, and
*substantial scale*. Following these principles, we curated a dataset
comprising billions of videos and images. This section offers a detailed
introduction to the data construction pipeline employed for .

## Pre-training Data

We curated and deduplicated a candidate dataset sourced from both
internal copyrighted sources and publicly accessible data. In the
pre-training stage, our goal is to select high-quality and diverse data
from this expansive yet noisy dataset to facilitate effective training.
Throughout the data mining process, we designed a four-step data
cleaning process, focusing on fundamental dimensions, visual quality,
and motion quality. Subsequently, we will also highlight our data
processing workflow for constructing visual text data.

**Fundamental dimensions.** The fundamental dimensions of our data
filtering framework focus on the intrinsic attributes of the source
video and image data, enabling efficient preliminary filtering out of
all the unsuitable data. Specifically, our multidimensional filtering
approach encompasses the following critical aspects: *Text detection*. A
lightweight OCR detector is implemented to quantify text coverage
ratios, effectively excluding videos and images with excessive textual
elements to maintain visual clarity. *Aesthetic evaluation*: We use the
widely adopted LAION-5B  aesthetic classifier to perform an initial
quality assessment of our images, quickly filtering out low-quality
data. *NSFW Score*. Through our internal safety assessment model, we
systematically evaluate and filter inappropriate content based on
computed NSFW scores in all training data. *Watermark and logo
detection*. We detect whether the video or images contain watermarks and
logos, and crop these elements during training. *Black border
detection*. Utilizing heuristic-based detection methods, we
automatically crop extraneous black borders to maintain focus on
content-rich regions. *Overexposure detection*. Our trained expert
classifier evaluates and filters out data with abnormal tonal
distributions, ensuring optimal visual quality in the training dataset.
*Synthetic image detection*. Empirical evidence indicates that even
minimal contamination ($12 grid
representation to reduce computation.

For the video, we sample 3 frames per second, with an upper limit of 129
frames. To further reduce computation, we employ a slow-fast encoding
strategy: maintaining the original resolution for every fourth frame and
applying global average pooling to the embeddings of the remaining
frames. Experiments on long video benchmarks indicate that the slow-fast
mechanism enhances understanding with a limited number of visual tokens
(*e.g.*, improving performance from 67.6% to 69.1% on VideoMME ) without
subtitles.

**Training.** Following recent state-of-the-art methods, our training
process is divided into three stages. In the first stage, we freeze the
ViT and LLM, and train only the multi-layer perceptron to align the
visual embeddings with the LLM input space, using a learning rate of
1e-3. In the second stage, all parameters are made trainable. In the
final stage, we conduct end-to-end training on a small set of
high-quality data. For the last two stages, the learning rates are set
to 1e-5 for the LLM and MLP, and 1e-6 for the ViT.

### Evaluation

Automatic evaluation is crucial for the development of our caption
model, allowing us to identify the strengths and weaknesses of each
model version. Following CAPability , we develop an automatic caption
evaluation pipeline. We focus on ten key visual dimensions in video
generation: action, camera angle, camera motion, object category, object
color, object counting, OCR, scene, style, and event. For each
dimension, we randomly sampled 1,000 videos and their corresponding
captions generated by our model. We also used Google Gemini 1.5 Pro to
generate dense video captions for this dataset. We evaluate our captions
and Gemini-generated captions by calculating the F1 metrics for each
dimension following CAPability . The results are illustrated in
Fig. 4. Our caption performs better in
video event, camera angle, camera motion, style, and object color, while
Gemini excels in action, OCR, scene, object category, and object
counting.

# Model Design and Acceleration

## Spatio-temporal Variational Autoencoder



Variational Autoencoders (VAEs)  play a crucial role in learning compact
latent representations from high-dimensional visual data (especially
videos), facilitating scalable and efficient training of generative
models, *e.g.*, diffusion models. However, designing effective VAEs for
video generation tasks faces several challenges. First, videos
inherently possess both spatial and temporal dimensions, requiring the
VAE to capture complex spatio-temporal dependencies. Second, the
inherent high-dimensional nature of video (*e.g.*, multiple frames with
high-resolution pixels) increases memory consumption and computational
costs, making it difficult to scale VAEs to long video sequences. Third,
ensuring temporal causality (*i.e.*, future frames do not influence past
frames) is critical for generating realistic and coherent video content,
yet this constraint introduces additional architectural complexities.

To address these challenges, we propose a novel 3D causal VAE
architecture (-VAE) specifically designed for video generation. We
combine multiple strategies  to improve spatio-temporal compression,
reduce memory usage, and ensure temporal causality. These enhancements
make -VAE more efficient and scalable and better suited for integration
with diffusion-based generative models such as DiT. Next, we introduce
the network design, training details, efficient inference, and
experimental comparison of our -VAE.

### Model Design

To achieve a bidirectional mapping between the high-dimensional pixel
space and the low-dimensional latent space, we design a 3D causal VAE as
illustrated in Fig. 5. Given an input video
$V\in \mathbb{R}^{(1+T) \times H \times W \times 3}$, -VAE compresses
its spatio-temporal dimensions to $[1+T/4, H/8, W/8]$ while expanding
the number of channels $C$ to 16. Specifically, the first frame is only
spatially compressed to better handle the image data, following
MagViT-v2 . In terms of architecture, we replace all GroupNorm layers 
with RMSNorm layers  to preserve temporal causality. This modification
enables the use of the feature cache mechanism (refer to
Sec. 4.1.3), significantly improving the
inference efficiency. Furthermore, we halve the input feature channel in
the spatial upsampling layer, resulting in a 33% reduction in memory
consumption during inference.

By carefully tuning the number of base channels further, -VAE achieves a
compact model size of only 127M parameters. This optimization reduces
the encoding time and memory usage, thus benefiting the training of
subsequent diffusion transformer models.

### Training

We adopt a three-stage approach to train -VAE. First, we construct a 2D
image VAE with the same structure and train it on image data. Then, we
inflate the well-trained 2D image VAE into a 3D causal -VAE to provide
an initial spatial compression prior, which drastically improves the
training speed compared to training a video VAE from scratch. At this
stage, -VAE is trained on the low-resolution (128$\times$128)
and small frame number (5-frame) videos to accelerate convergence speed.
The training losses include $L1$ reconstruction loss, KL loss, and LPIPS
perceptual loss, which are weighted by coefficients of 3, 3e-6, and 3,
respectively. Finally, we fine-tune the model on high-quality videos
with different resolutions and frame numbers, integrating the GAN loss 
from a 3D discriminator.



### Efficient Inference

To efficiently support the encoding and decoding of arbitrarily long
videos , we implement a feature cache mechanism within the causal
convolution module of -VAE.

Specifically, the number of video sequence frames follows the $1+T$
input format, so we divide the video into $1+T/4$ chunks, which is
consistent with the number of latent features. When processing input
video sequences, the model employs a chunk-wise strategy where each
encoding and decoding operation handles only the video chunk
corresponding to a single latent representation. Based on the temporal
compression ratio, the number of frames in each processing chunk is
limited to at most 4, effectively preventing memory overflow.

To ensure temporal continuity between context chunks, our model
strategically maintains frame-level feature caches from preceding
chunks. These cached features are systematically integrated into the
causal convolution computations of subsequent chunks. The specific
process is illustrated in
Fig. 6, which demonstrates two typical
scenarios in our feature cache mechanism. In
Fig. 6 (a), in our default setting, the
causal convolution does not change the number of frames, and we need to
maintain two cached features (convolution kernel size is 3) from
historical frames. For the initial chunk, we apply zero padding with two
dummy frames to initialize the cache buffer. Subsequent chunks
systematically reuse the last two frames from the preceding chunk as
cached features while discarding obsolete historical data.
Fig. 6 (b) presents the scenario involving
2$\times$ temporal downsampling (stride is equal to 2). Here, the
scenario necessitates different cache management: we implement
single-frame cache padding exclusively for non-initial chunks to ensure
dimensional consistency. This configuration guarantees that the output
sequence length follows the exact downsampling ratio while maintaining
causal relationships across the boundaries of the chunk.

This feature cache mechanism not only optimizes memory utilization but
also preserves feature coherence across chunk boundaries, thereby
supporting stable inference for infinite-length videos.

### Evaluation



**Quantitative results.** This study conducts a comprehensive evaluation
of the performance of various state-of-the-art (SOTA) video VAE models 
by analyzing their Peak Signal-to-Noise Ratio (PSNR) and processing
efficiency (frame per unit latency (second)). Notably, for fair
comparison, most models  adopt the same compression rate and latent
feature dimension as our approach, *i.e.,* a compression rate of
$4\times8\times8$ with a latent dimension of 16. Open Sora Plan  has the
same compression ratio as ours, but the latent dimension is 4. SVD 
employs a compression rate of $1\times8\times8$ with a latent dimension
of 4, Step Video  employs a compression rate of $8\times16\times16$ with
a latent dimension of 64, and Mochi  uses a compression rate of
$6\times8\times8$ with a latent dimension of 12. The evaluation is
performed on 200 videos. These videos consist of 25 frames, and the
resolution is set to $720 \times 720$. In
Fig. 7, the size of the circles is
positively correlated with the number of model parameters.

Experimental results indicate that -VAE exhibits highly competitive
performance across both metrics, showcasing a compelling dual advantage
of superior video quality and high processing efficiency. It is worth
noting that under the same hardware environment, our VAE’s
reconstruction speed is 2.5 times faster than the existing SOTA method
(*i.e.*, HunYuan Video). This speed advantage will be further
demonstrated at higher resolutions due to the small size design of our
VAE model and the feature cache mechanism.

This advancement offers an efficient solution for video reconstruction
tasks and video generation training. These findings not only validate
the effectiveness of the proposed model design but also provide valuable
insights for the future development of VAE technologies.



**Qualitative results.** To validate the video reconstruction
performance of our method across diverse scenarios,
Fig. 8 presents visual results for
texture, face, text, and high-motion scenes. Compared to existing VAE
models, -VAE demonstrates superior performance in these contexts. In the
texture scene shown in the first row, our method is capable of capturing
details more accurately, such as the texture and direction of hair. In
the facial scene depicted in the second row, -VAE preserves facial
features while reducing blurring and distortion around the lips. In the
text scene illustrated in the third row, our method successfully
restores textual content with clarity, mitigating issues of character
distortion and loss. In the high-motion scene presented in the fourth
row, -VAE maintains the motion sharpness of video frames. These results
indicate that our method offers significant advantages in handling a
variety of complex scenarios.



## Model Training

In this section, we will provide a detailed introduction to the
architecture design of the foundational video model, specifically for
the text-to-video task, along with the details of the pre-training and
post-training phases.

In Fig. 9, we present the architecture
of , which is designed based on the current mainstream DiT 
architecture, which typically consists of three primary components:
-VAE, diffusion transformer, and text encoder. For a given video
$V\in \mathbb{R}^{(1+T) \times H \times W \times 3}$, the encoder of
-VAE encodes it from pixel space into the latent space
$x \in \mathbb{R}^{1+T/4 \times H/8 \times W/8}$, which is then fed into
the subsequent diffusion transformer structure.

### Video Diffusion Transformer



r0.4 



**Diffusion transformer**. The diffusion transformer mainly consists of
three components: a patchifying module, transformer blocks, and an
unpatchifying module. Within each block, we focus on effectively
modeling spatio-temporal contextual relationships and embedding text
conditions alongside time steps. In the patchifying module, we use a 3D
convolution with a kernel size of $(1, 2, 2)$ and apply a flattening
operation to convert $x$ into a sequence of features with the shape of
$(B, L, D)$, where $B$ denotes the batch size,
$L=(1+T/4) \times H/16 \times W/16$ represents the sequence length, and
$D$ indicates the latent dimension. Specifically, as illustrated in
Fig. [fig:dit_block], we employ the
cross-attention mechanism to embed the input text conditions, which can
ensure the model’s ability to follow instructions even under
long-context modeling. Additionally, we employ an MLP with a Linear
layer and a SiLU  layer to process the input time embeddings and predict
six modulation parameters individually. This MLP is shared across all
transformer blocks, with each block learning a distinct set of biases.
Through extensive experiments, we have demonstrated that this design can
reduce the parameter count by approximately $25\%$ and reveal a
significant performance improvement with this approach at the same
parameter scale.

**Text encoder**. ’s architecture uses the umT5  to encode input text.
Through extensive experiments, we find that umT5 has several advantages
in our framework: 1) It possesses strong multilingual encoding
capabilities, allowing it to understand both Chinese and English
effectively, as well as input visual text; 2) Under the same conditions,
we find that umT5 outperforms other unidirectional attention mechanism
LLMs in composition; 3) It exhibits superior convergence, with umT5
achieving faster convergence at the same parameter scale. Based on these
findings, we ultimately chose umT5 as our text embedder.

### Pre-training

We leverage the flow matching framework  to model a unified denoising
diffusion process across both image and video domains. We first conduct
pre-training on low-resolution images, followed by multi-stage joint
optimization of images and videos. The image-video joint training
evolves using progressively upscaled data resolutions and extended
temporal durations throughout the stages.

**Training objective.** Flow matching provides a theoretically grounded
framework for learning continuous-time generative processes in diffusion
models. It circumvents iterative velocity prediction, enabling stable
training via ordinary differential equations (ODEs) while maintaining
equivalence to maximum likelihood objectives. During training, given an
image or video latent $x_{1}$, a random noise
$x_{0} \sim \mathcal{N}(0, I)$, and a timestep $t \in[0,1]$ sampled from
a logit-normal distribution, an intermediate latent $x_{t}$ is obtained
as the training input. Following Rectified Flows (RFs) , $x_{t}$ is
defined as a linear interpolation between $x_{0}$ and $x_{1}$,
$$\label{x_t}
    x_{t}=t x_{1}+(1-t) x_{0}.$$ The groud truth velocity $v_{t}$ is
$$\label{v_t}
v_{t}=\frac{d x_{t}}{d t}=x_{1}-x_{0}.$$ The model is trained to predict
the velocity, thus, the loss function can be formulated as the mean
squared error (MSE) between the model output and $v_{t}$,

$$\label{rf_loss}
\mathcal{L} = \mathbb{E}_{x_0, x_1, c_{txt}, t}||u(x_t, c_{txt}, t; \theta)-v_t||^2,$$
where $c_{txt}$ is the umT5 text embedding sequence of 512 tokens long,
$\theta$ is the model weights, and $u(x_t, c_{txt}, t; \theta)$ denotes
the output velocity predicted by the model.

**Image pre-training.** Our experimental analysis reveals two critical
challenges in direct joint training with high-resolution images and
long-duration video sequences: (1) Extended sequence lengths (typically
$81$ frames for $1280\times720$ video) substantially reduce training
throughput. Under fixed GPU-hour budgets, this results in insufficient
data throughput, thereby impeding model convergence rates. (2) Excessive
GPU memory consumption forces suboptimal batch sizes, inducing training
instability caused by spikes in gradient variance. To mitigate these
issues, we initialize the 14B model training through low-resolution
($256~px$) text-to-image pre-training, enforcing cross-modal
semantic-textual alignment and geometric structure fidelity before
progressively introducing high-res video modalities.

**Image-video joint training.** Following large-scale $256~px$
text-to-image pre-training, we implement staged joint-training with
image and video data through a resolution-progressive curriculum. The
training protocol comprises three distinct stages differentiated by
spatial resolution areas: (1) In the first stage, we conduct joint
training with $256~px$ resolution images and 5-second video clips
($192~px$ resolution, $16$ fps). (2) In the second stage, we initiate
spatial resolution scaling by upgrading both image and video resolutions
to $480px$ while maintaining a fixed 5-second video duration. (3) In the
final phase, we escalate spatial resolution to $720 px$ for both images
and 5-second video clips.

**Pre-training configurations.** We employ efficient training at
bf16-mixed precision combined with the AdamW  optimizer with a weight
decay coefficient of $1e^{-3}$. The initial learning rate is set to
$1e^{-4}$, and dynamically reduced triggered by plateaus of the FID and
CLIP Score metrics.

### Post-training

In the post-training stage, we maintain the same model architecture and
optimizer configuration from the pre-training stage, initializing the
network with the pre-trained checkpoint. We conduct joint training at
resolutions of $480 px$ and $720 px$ using the post-training video
dataset detailed in
Sec. 3.2.

## Model Scaling and Training Efficiency

### Workload Analysis

In , the computational costs associated with the text encoder and VAE
encoder are lower than those of the DiT model, which accounts for more
than 85% of the overall computation during training. For the DiT model,
the main computational cost is given by the expression
$L(\alpha bsh^2+\beta bs^2h)$, where $L$ denotes the number of DiT
layers, $b$ is the micro batch size, $s$ indicates the sequence length,
and $h$ denotes the hidden dimension size. Here, $\alpha$ corresponds to
the cost of the linear layers and $\beta$ to that of the attention
layer, for the non-causal attention used in , $\beta$ is 4 for the
forward pass and 8 for the backward pass. Unlike general LLM models, the
sequence length $s$ processed by often reaches hundreds of thousands or
more. Since the computational cost of attention increases quadratically
with the number of tokens, while the cost associated with the linear
layers increases only linearly, the attention mechanism gradually
becomes the training bottleneck. In scenarios where the sequence length
reaches 1 million, the computation time for attention can account for up
to 95% of the end-to-end training time.

During the training process, only the DiT model is optimized while the
text encoder and the VAE encoder remain frozen. Therefore, GPU memory
usage is concentrated on training the DiT. The GPU memory usage for the
DiT can be expressed as $\gamma Lbsh$, where $\gamma$ depends on the
implementation of the DiT layers, L denotes the number of DiT layers.
Because video tokens, input prompts, and timesteps are taken as model
input, the value of $\gamma$ is generally larger than in ordinary LLM
models. For example, while standard LLMs might have a $\gamma$ of 34 ,
the DiT model’s $\gamma$ can exceed 60. Consequently, when the sequence
length reaches 1 million tokens and the micro-batch size is 1, the total
GPU memory usage for activations in a 14B DiT model can exceed 8 TB.

In summary, the primary computational cost in arises from the attention
mechanism. While the computational cost increases quadratically with the
sequence length $s$, the GPU memory usage scales only linearly with $s$.
This provides a valuable reference for guiding subsequent optimization
efforts.

### Parallelism Strategy

The model mainly consists of three modules: VAE, text encoder, and DiT.
Following the cache mechanism optimization outlined in
Sec. 4.1.3, the VAE module exhibits
minimal GPU memory usage and can seamlessly adopt Data Parallelism (DP).
In contrast, the text encoder requires over 20 GB of GPU memory,
necessitating the use of model weight sharding to conserve memory for
the subsequent DiT module. To address this, we employ a Data Parallel
(DP) approach combined with Fully Sharded Data Parallel (FSDP)  
strategy. The model parameters, gradients, and optimizer states of the
DiT will exceed the capacity of a single GPU, with activation storage
reaching approximately 8 TB in a scenario involving 1M tokens with a
batch size of 1. Moreover, the DiT portion accounts for a significant
proportion of the overall computational workload. Thus, we focus on
developing efficient distributed strategies specifically for the DiT
module.

**DiT parallel strategy.** In light of the DiT model’s modest size, and
considering the optimization of overlapping computation and
communication as well as maximizing the size of data parallelism (DP),
we have selected FSDP as our parameter sharding strategy.

Regarding activations, the input shape of a DiT block is $[b, s, h]$,
where the $b$ dimension corresponds to data parallelism, $s$ represents
the sequence length, and $h$ denotes the hidden dimension. Therefore, we
need to examine sharding strategies for the $s$ and $h$ dimensions.
Sharding along the s dimension is achieved through Context Parallelism
(CP), with common methods including Ulysses  and Ring Attention .
Sharding along the $h$ dimension primarily involves Megatron’s tensor
parallelism (TP)  combined with sequence parallelism (SP) , which shards
the hidden dimension of the activations by splitting the weights. Since
the communication overhead of CP is smaller than that of (TP $+$ SP), we
propose using CP to accelerate the batch-level computations while
reducing the activation memory footprint on each GPU. We have designed a
two-dimensional (2D) CP that combines the characteristics of Ulysses and
Ring Attention, similar to USP . In this design, the outer layer employs
Ring Attention, and the inner layer leverages Ulysses. This combination
mitigates the slow cross-machine communication inherent in Ulysses and
addresses the need for large block sizes after sharding in Ring
Attention, thereby maximizing the overlap between outer-layer
communication and inner-layer computation. In scenarios with a sequence
length of 256K and 16 GPUs across 2 machines, the communication overhead
of 2D Context Parallelism is reduced from over $10\%$ with Ulysses to
below $1\%$.



The FSDP group and CP group intersect within the FSDP group, the DP size
equals the FSDP size divided by the CP size. After satisfying memory and
single-batch latency requirements, we use DP for the scaling.
Fig. 10 illustrates the distributed
scheme for the DiT module. For example, in a configuration with 128
GPUs, the CP size is 16, with Ulysses set to 8 and Ring Attention to 2;
FSDP is set to 32, corresponding to a batch size of 2b; DP is set to 4,
resulting in a global batch size of 8b.

**Distributed strategy switching for different modules.** During
training, different modules utilize the same resources. Specifically, we
apply DP to the VAE and Text Encoder, while the DiT module employs a
combination of DP and CP. Therefore, when the outputs of the VAE and
Text Encoder are fed into the DiT module during training, it is
necessary to switch distributed strategies to avoid resource wastage.
Specifically, CP requires that devices within the CP group read the same
batch of data. Yet, if devices in the VAE and Text Encoder sections also
read the same data, this leads to redundant computations. To eliminate
this redundancy, devices within the CP group initially read different
data. Then, before CP, we perform a loop traversal of size equal to the
CP size, sequentially broadcasting the data read by different devices
within the CP group to ensure that the inputs within CP are identical.
In this way, the time proportion of VAE and Text Encoder within a single
model iteration is reduced to $1 / CP$, thereby enhancing overall
performance.

### Memory Optimization

As introduced, the computational cost in  grows quadratically with
sequence length $s$, whereas GPU memory usage scales linearly with $s$.
This means that in long sequence scenarios, computation time can
eventually exceed the PCIe transfer time required for offloading
activations. Specifically, the PCIe transfer time for offloading the
activation of one DiT layer can be overlapped with the computation of
just 1 to 3 DiT layers. Compared to gradient checkpointing (GC)
strategies , activation offloading  that allows computation overlap
provides an effective way to reduce GPU memory usage without sacrificing
end-to-end performance. Therefore, we prioritize activation offloading
to reduce GPU memory. Since CPU memory is also prone to exhaustion in
long sequence scenarios, we combine offloading with GC strategies,
prioritizing gradient checkpointing for layers with high GPU
memory-to-computation ratios.

### Cluster Reliability

By leveraging Alibaba Cloud’s advanced intelligent scheduling, slow
machine detection, and robust self-healing capabilities within the
training cluster, high stability is ensured. Hardware issues are
identified during the job startup phase, ensuring that only healthy
nodes are allocated to training tasks. Throughout training, any faulty
nodes are promptly isolated and repaired, with tasks automatically
restarted and training seamlessly resumed. This efficient orchestration
effectively combines reliability with high performance, ensuring overall
system stability.

## Inference

The primary objective of inference optimization is to minimize the
latency of video generation. Unlike training, the inference process
involves multiple sampling steps, typically around 50. Accordingly, we
employ techniques such as quantization and distributed computing to
reduce the time required for each individual step. Additionally, we
leverage the attention similarities between steps to decrease the
overall computational load. Furthermore, for classifier-free guidance
(CFG) , we exploit its inherent similarities to further minimize
computational requirements.

### Parallel Strategy



r0.5 



As discussed in Sec. 4.3.2, we utilize Context Parallel to
reduce the latency of generating a single video when scaling to multiple
GPUs. Additionally, for large models like 14B, we adopt model sharding
to mitigate GPU memory constraints. *Model Sharding Strategy*: Given the
long sequence lengths during inference, FSDP incurs less communication
overhead compared to TP and allows for computation overlap. Therefore,
we adopt the FSDP method for model sharding, consistent with our
training approach. *Context Parallel Strategy*: We employ the same 2D
Context Parallelism as that in the training stage, with RingAttention
serving as the outer loop and Ulysses operating as the inner loop.
Equipped with 2D Context Parallel and FSDP parallel strategy, DiT
achieves nearly linear speedup on the 14B model, as shown in the
Fig. [fig:scaling].

### Diffusion Cache

We conduct a thorough analysis of the model and identify the following
characteristics in its inference process:

- Attention similarity: Within the same DiT block, attention outputs
  across different sampling steps exhibit significant similarity.

- CFG similarity: In the later stages of sampling, there is a notable
  similarity between conditional and unconditional DiT outputs.

These findings are also highlighted in recent works, such as
DiTFastAttn  and FasterCache . We leverage these characteristics to
implement diffusion caching to reduce the computational workload. In
practice, we tailor the diffusion cache to the model to ensure lossless
performance. Specifically, we use the validation set to select certain
steps for caching:

- Attention cache: For the attention component, we perform an attention
  forward pass every few steps and cache the results, reusing the cached
  results for the other steps.

- CFG cache: For the unconditional part, we perform a DiT forward pass
  every few steps and reuse the conditional results for the other steps.
  Additionally, we apply residual compensation methods similar to
  FasterCache to prevent the degradation of fine details.

After applying diffusion caching to the 14B text-to-video model, we have
improved the inference performance by 1.62×.

### Quantization

**FP8 GEMM.** We found that using FP8 precision for the GEMM operations,
combined with per-tensor quantization on the weights and per-token
quantization on the activations in the sampling steps, incurs minimal
performance loss. Thus we apply FP8 quantization to all GEMM operations
in the DiT block. The FP8 GEMM delivers twice the performance of BF16
GEMM and achieves a 1.13× speedup in the DiT module.

**8-Bit FlashAttention.** Although FlashAttention3  achieves high
performance, its native FP8 implementation suffers from significant
quality degradation in video generation, as observed in our experiments.
Conversely, SageAttention  employs a mixed precision of int8 and fp16 to
reduce precision loss, but it does not provide specific adaptations for
Hopper GPUs. As of this report’s publication, subsequent optimizations
to SageAttention have demonstrated improved Hopper compatibility. Future
work will explore the integration of these performance enhancements.

We present our optimizations applied to FA3-FP8 implementation, focusing
on both numerical stability and computational efficiency.

**Accuracy.** To mitigate the numerical errors in attention observed in
under FA3-FP8 quantization, we apply two primary techniques:

- Mixed 8-Bit optimization. The native implementation of FA3 uses FP8
  (E4M3) for all input tensors ($Q$, $K$, $V$). Based on both our
  analysis of the data distribution and the empirical evidence from the
  SageAttention, we adopt a hybrid quantization strategy: Int8 for
  $S=QK^T$, FP8 for $O=PV$.

- FP32 accumulation for cross-block reduction. The native FP8 WGMMA
  implementation employs $~$14-bit accumulators, making it prone
  to overflow during long-sequence processing. To address this
  limitation, our approach leverages FP8 WGMMA for intra-block $PV$
  reduction while implementing cross-block accumulation through CUDA
  cores with Float32 registers, a strategy inspired by DeepSeek-V3’s 
  FP8 GEMM methodology.

**Performance.** To maintain kernel performance under mixed INT8-FP8
quantization while enhancing the accumulation precision of $O=PV$, we
employ two techniques:

- Fuse Float32 accumulation with intra-warpgroup pipelining.
  FlashAttention3 employs an intra-warpgroup pipeline that overlaps
  WGMMA with softmax and scaling. To mitigate the performance penalty
  introduced by Float32 $PV$ accumulation across blocks, we fuse the
  Float32 accumulation with $rescale\_o$ operation.

- Block size tuning. Due to the additional register pressure introduced
  by Float32 $PV$ accumulation register across blocks, we adjust the
  block size to reduce register spilling. After optimization, our 8-bit
  FlashAttention achieves 95% MFU on NVIDIA H20 GPUs.

Consequently, with 8-bit FlashAttention, we boost the inference
efficiency by more than 1.27×.



| User prompts                                                                                                                                   | Rewritten results                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|:-----------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A Viking warrior wields a great axe with both hands, battling a mammoth at dusk, amidst a snowy landscape with snowflakes swirling in the air. | An epic battle scene, unfolds as a tall and muscular Viking warrior wields a heavy great axe with both hands, facing off against a massive mammoth. The warrior is clad in leather armor and a horned helmet, with prominent muscles and a fierce, determined expression. The mammoth is covered in long hair, with sharp tusks, and roars angrily. It is dusk, and the snowy landscape is filled with swirling snowflakes, creating an intense and dramatic atmosphere. The backdrop features a barren ice field with the faint outlines of distant mountains. The use of cool-toned lighting emphasizes strength and bravery. The scene is captured in a dynamic close-up shot from a high-angle perspective. |
| The camera follows a motorboat chasing dolphins in the sea.                                                                                    | In a documentary photography style, the camera follows a motorboat chasing a pod of dolphins leaping out of the vast ocean. On the motorboat, there is a driver wearing a life jacket and a safety helmet, with a focused and excited expression. The dolphins are sleek and smooth-skinned, occasionally leaping out of the water and diving back in with agility. The sky is bright blue, the sun is shining, and the sea is shimmering with light. A few white clouds dot the distant sky. The scene is dynamic and full of energy, captured from a medium shot in a tracking perspective.                                                                                                                   |
| The tiny Potato King, wearing a majestic crown, sits on the throne as potato subjects pay homage to it.                                        | In a surrealist style, the tiny Potato King wears a magnificent gold crown and sits on a towering throne. Its skin has a natural earthy yellow tone with subtle bumpy textures. The potato subjects are lined up on either side, bowing their heads in homage to the king. The background features the grand interior of a palace, with gold and red decorations that appear luxurious and solemn. A beam of light shines down from above, creating a sacred atmosphere. The scene is captured in a close-up shot from a high-angle perspective.                                                                                                                                                                |



## Prompt Alignment

Prompt alignment is designed to boost the effectiveness of model
inference by aligning the user’s input prompt with the format and style
of captions used during training. We employ two primary strategies in
this process. First, we augment each video with a variety of captions,
thereby increasing diversity. Second, we rewrite user prompts to align
them with the distribution of video captions from the training stage. In
the following, we detail these strategies.

To accommodate diverse user inputs, each image or video is paired with
multiple captions reflecting various styles and lengths. For example, we
use captions of differing lengths (e.g., long, medium, and short) and
styles (e.g., formal, informal, and poetic) to create a varied set of
sample-caption pairs for our training data. These captions establish
different mapping relations between text and video, effectively covering
the range of prompts that users might provide.

Despite the diversity of captions, a distribution mismatch often exists
between the dataset captions and user input prompts. Typically, users
prefer concise prompts consisting of a few simple words, which are
significantly shorter than the training captions. This discrepancy can
adversely affect the quality of the generated videos. To address this
issue, we rewrite prompts by utilizing a large language model (LLM) to
refine user prompts. Our primary objective is to align the distribution
of these refined prompts with the distribution of the training captions,
thereby achieving improved inference performance. The LLM is guided by
the following principles when rewriting user prompts:

1.  We instruct the LLM to add details to prompts without altering their
    original meanings, enhancing the completeness and visual appeal of
    the generated scenes.

2.  The rewritten prompts should incorporate natural motion attributes,
    where we add appropriate actions for the subject based on its
    category to ensure smoother and more fluent motion in the generated
    videos.

3.  We recommend structuring the rewritten prompts similarly to the
    post-training captions, beginning with the video style, followed by
    an abstract of the content, and concluding with a detailed
    description. This method helps align prompts with the distribution
    of high-quality video captions.

We evaluate the effectiveness of LLM-assisted prompt rewriting on our
benchmark, and several results are presented in
Tab. [tab:rewritten_prompts]. Our
findings suggest that LLMs with strong instruction-following
capabilities can generate suitable prompts that enhance video
generation. To balance speed and performance, we use Qwen2.5-Plus as our
rewriting model.



## Benchmarks

Existing video generation evaluation methods, such as Fréchet Video
Distance (FVD)  and Fréchet Inception Distance (FID) , lack alignment
with human perception. We propose an automated, comprehensive, and
human-aligned -Bench for evaluating video generation models.

-Bench is composed of three core dimensions: dynamic quality, image
quality, and instruction following with 14 fine-grained metrics. We
design specific scoring algorithm for each dimension, using traditional
detectors for the simple tasks (, object detection) and multi-modal
Large Language Models (MLLMs) for the complex tasks.

**Dynamic quality**. Dynamic quality aims to evaluate the quality and
stability of the model in non-static scenarios.

*Large motion generation.* To measure the upper limit of motion
generation, prompts are designed to encourage significant motions. We
calculate the optical flow of the generated video using RAFT  and assess
the motion scores by the magnitude of the normalized optical flow.

*Human artifacts.* Existing evaluation methods lack the ability to
detect AI-generated artifacts. Thus, we train a YOLOv3-based model  on
20,000 AI-generated images annotated by humans to identify the locations
of artifacts. The artifact scores take into account the predicted
likelihood, the bounding boxes, and the corresponding duration.

*Physical plausibility & smoothness*. To evaluate physical plausibility,
we design prompts to generate physics-related movements (, ball
bouncing, object interactions, water flow) and employ Qwen2-VL ,
leveraging its extensive real-world physics knowledge. Through a video
question-answer model, we detect violations of physical laws, such as
object clipping, unrealistic collisions, or gravity-defying movements.
To evaluate smoothness, prompts are crafted to include complex motions,
with Qwen2-VL identifying artifacts to assess motion fluidity.

*Pixel-level stability*. Pixel-level stability is used to determine if
noise frequently appears in the video. We employ optical flow to
identify static regions and calculate frame-wise differences within
those areas. For stable videos, these differences within static regions
are relatively minimal.

*ID consistency.* Three kinds of sub-dimensions are considered: human
consistency, animal consistency, and object consistency covering
different collections of prompts. Frame-level DINO features are
extracted to compare the frame-wise similarity as the consistency score.

**Image quality.** The purpose of the image quality dimension is to
evaluate the visual quality and aesthetic perception.

*Comprehensive image quality.* Measuring the visual quality requires an
evaluation of the fidelity and aesthetics. For fidelity, we employ
MANIQA , which effectively detects blur and artifacts, ensuring precise
clarity assessment. For aesthetics, we utilize two established models:
the LAION-based aesthetic predictor  and MUSIQ . We compute the final
image quality score by averaging these three evaluators.

*Scene generation quality.* We evaluate two key aspects: frame-wise
scene consistency and scene-text alignment. For consistency, we measure
CLIP similarity across consecutive frames to ensure temporal coherence.
For alignment, we compute CLIP similarity between frames and their
corresponding text to verify semantic accuracy. We compute the quality
score by weighted averaging these metrics.

*Stylization.* We employ Qwen2-VL to evaluate the artistic video
generation capability through frame-level question answering.

**Instruction following.** Instruction following assesses the model’s
ability to comprehend and execute textual instructions.

*Single object & multiple objects & spatial positions.* We employ
Qwen2-VL to predict object categories, counts, and spatial relationships
within videos. The evaluation score represents the average number of
frames that accurately correspond to the given prompts.

*Camera control.* We evaluate five camera movement styles: panning,
lifting, zooming in/out, aerial shots, and tracking shots by 130
tailored prompts. For panning, lifting, and zooming, we analyze optical
flow via RAFT to assess movement style. For complex aerial and tracking
shots, we employ Qwen2-VL to infer camera movement through video
question-answering.

*Action instruction following.* We categorize motions into human (,
running), animal (, crawling), and object movements (, flying). As
evaluating these motions requires prior knowledge of their execution, we
employ Qwen2-VL by providing it with keyframes. We query the model for
alignment of the action, completion of the action, and presence of
artifacts to comprehensively evaluate alignment between text and video.

**Human feedback guided weighting strategy.** We utilize human feedback to assign a
comprehensive video quality score, considering that each dimension has a
different impact on user preferences rather than simply averaging them.
We collect over 5,000 pairwise comparisons of videos generated by
different models including . Users evaluate pairs based on a given text,
indicating their preference and assigning approximate scores. The
dimension preference weighting is derived from the Pearson correlation
between model-generated scores and human-rated scores, serving as a
weighting factor in the final score calculation.

## Evaluation

### Metrics and Results

**Baselines and metrics.** As of the time of writing, there are numerous
comparative competitors in the market, including commercial models such
as Kling , Hailuo , Sora , Runway and Vidu , as well as open-source
models like Mochi , CogVideoX and Hunyuan . In this section, we compare
our model with the commercial and open-source models that have been
evaluated through benchmarks and received relatively positive user
feedback.

**Quantitative results.** We collected 1,035 samples for each candidate
model using -Bench prompts and conducted fair evaluations by scoring the
models with -Bench. The evaluation focused on three key metrics: dynamic
quality, image quality, and instruction-following accuracy. We rank the
performance of the different generation models by calculating total
scores weighted according to human preferences
(Sec. [hf-align]). The results in Table
[tab:model_performance]
indicate that outperforms existing commercial and open-source models.

**Qualitative results.** As illustrated in the
Fig. 12,  effectively generates diverse,
high-quality videos directly from textual descriptions.  excels at
synthesizing dynamic scenes involving large-scale, complex movements, as
well as scenarios reflecting physical interactions. Moreover, it adeptly
handles various artistic styles and consistently produces
cinematic-quality visuals. Additionally,  demonstrates robust
multilingual text-generation capabilities, integrating Chinese and
English texts into animations and giving rise to movie-level textual
effects.



**Human evaluation.** We design over 700 evaluation tasks, which are
annotated by more than 20 individuals and assessed by four key
dimensions: alignment, image quality, dynamic quality, and overall
quality. The results in
Table 1 demonstrate that in the T2V
task, the 14B model consistently excels across all visual quality
dimensions, indicating its superior performance.



|                 | CN-TopA | CN-TopB | CN-TopC | Runway | **All Rounds** |
|:----------------|:-------:|:-------:|:-------:|:------:|:--------------:|
| Visual Quality  |  30.6%  |  15.9%  |  27.8%  | 48.1%  |      5710      |
| Motion Quality  |  16.1%  |    %    |  14.9%  | 40.3%  |      5785      |
| Matching        |  46.0%  |  57.9%  |  56.7%  | 69.1%  |      5578      |
| Overall Ranking |  44.0%  |  44.0%  |  48.9%  | 67.6%  |      5560      |

Win rate gap of T2V.



  
*\** The table displays the proportion of instances in which the  14B
model outperformed other models in pairwise comparisons, relative to the
total number of comparisons.

**in public leaderboard.** has demonstrated state-of-the-art performance
on the VBench leaderboard , a widely used evaluation approach for video
generation task. The VBench employs a multidimensional assessment
architecture that decomposes video quality evaluation into 16 distinct,
human-aligned dimensions spanning aesthetic quality, motion smoothness,
and semantic coherence.

Our analysis focuses on two model variants: the 14B parameter 14B and
the 1.3B variant (shown in
Table [tab:vbench_res]). The 14B model
achieves a benchmark-leading performance with an aggregate score of
86.22%, including 86.67% in visual quality and 84.44% in semantic
consistency, significantly outperforming competitors like OpenAI’s Sora
and MiniMax’s Hailuo. The efficient 1.3B variant also remains
competitive, scoring 83.96% and surpassing commercial models such as
HunyuanVideo and Kling 1.0 , as well as the open-sourced CogVideoX1.5-5B
.

### Ablation Study

We provide ablation studies of key modules within our model design to
offer insights into the overall architecture’s contributions.
Experiments are conducted on 1.3B version for rapid evaluation.

**Ablation on adaptive normalization layers.** Given the substantial
parameter load of adaptive normalization layers  (adaLN) in DiT  , we
explore whether focusing on parameter volume in adaLN or increasing the
depth of the network layers is more effective. To adjust the parameter
volume, we follow the setup from PixArt  to study whether to share adaLN
or not. In the non-shared configuration, each block’s scale and shift
parameters are predicted by a block-specific MLP that takes as input the
time embedding. Instead, the shared AdaLN configuration (referred to as
AdaLN-single in PixArt) computes a single set of global parameters,
which involves predicting scale and shift values, in the first block,
which are then shared across all blocks. This sharing significantly
reduces the number of parameters.

We evaluate four configurations: (i) Full-shared-adaLN-1.3B: This is our
original model, where adaLN is fully shared across all 30 attention
blocks. (ii) Half-shared-adaLN-1.5B: AdaLN is shared in the first 15
attention blocks while not for the remaining blocks. This results in a
1.5B model with an increase of 0.2B parameters. (iii)
Full-shared-adaLN-1.5B (extended): AdaLN is shared across all attention
blocks, but the model depth is increased to 35 layers, keeping the 1.5B
parameter count. (iv) Non-shared-AdaLN-1.7B: AdaLN is not shared across
any blocks, and the model depth remains at 30 layers, resulting in a
1.7B model. We maintain other parameters unchanged and train the models
from scratch to perform text-to-image task for 200,000 steps (i.e., the
first stage of our text-to-video training), with the global batch size
of 1536. For comparison, we measure performance using the L2 loss
between generated images and real images in the latent space during the
training stage, with smaller training loss indicating better
convergence.



As presented in
Fig. 13, the Full-shared-AdaLN-1.3B
model, with fewer parameters, results in a slightly higher training loss
compared to others. When comparing Half-shared-AdaLN-1.5B and
Full-shared-AdaLN-1.5B, which have the same parameter count,
Full-shared-AdaLN-1.5B consistently achieves the lowest training loss.
This suggests that focusing on model depth rather than the parameters in
AdaLN leads to better performance. Additionally, the
Non-shared-AdaLN-1.7B model, despite having more parameters, does not
outperform the Full-shared-AdaLN-1.5B and further supports this idea.
Therefore, we adopt the fully shared AdaLN design to effectively reduce
parameter count while preserving performance.

**Ablation on text encoder.** We select three text encoders that can
handle bilingual inputs: umT5  (5.3B), Qwen2.5-7B-Instruct  , and
GLM-4-9B  . The T5 series has been a popular choice in video generation
models  , while the latter two encoders excel in language understanding,
especially among all LLM models under 10B parameters. For this ablation,
we keep other settings unchanged and train a text-to-image task with a
global size of 1536. Text embeddings are derived from the second-to-last
layers of Qwen2.5-7B-Instruct and GLM-4-9B. We measure the training loss
as shown in
Fig. 14.



r0.3

|           |           |       |
|:----------|:---------:|:-----:|
| Model     |    VAE    | VAE-D |
| 10k steps | **42.60** | 44.21 |
| 15k steps | **40.55** | 41.16 |



Compared to other powerful LLM-based encoders, umT5 demonstrates
superior text embedding performance. Notably, HunyuanVideo  highlights
that decoder-only LLMs use causal attention whereas umT5 adopts
bidirectional attention, making it suitable for diffusion models.
Following HunyuanVideo’s strategy, we incorporate a bidirectional token
refiner as an adapter layer. We apply this setup to both
Qwen2.5-7B-Instruct and GLM-4-9B, but the training losses and
visualizations still show umT5 performing most favorably. Further, we
compare umT5 with a pre-trained Multimodal Large Language Model, i.e.,
Qwen-VL-7B-Instruct . As shown in
Table [tab:ablation_textenc_fid],
using the second-to-last layer’s features from Qwen-VL results in
comparable generation performance to umT5, but with a larger model size.



**Ablation on autoencoder.** In addition to our VAE, we design a variant
called VAE-D, where the reconstruction loss is replaced by a diffusion
loss. We adopt both the pre-trained VAE and VAE-D on the text-to-image
generation task for 150,000 training steps until achieving convergence.
We measure the FID scores at both 100,000 and 150,000 steps. As shown in
Table [tab:ablation_vae_fid], VAE
model consistently achieves lower FID compared to VAE-D.

# Extended Applications

## Image-to-Video Generation

The image-to-video (I2V) generation task focuses on synthesizing a
dynamic video sequence from a static input image guided by a textual
prompt. This approach significantly enhances the controllability of
video generation by anchoring the output to a specific initial frame,
thereby garnering substantial interest within the research community.
Existing approaches, such as I2VGen-XL , SVD , and CogVideo , extend T2V
frameworks to I2V by channel-wise concatenation of the conditional
latent representation with the noise latent. Building upon this
paradigm, our -I2V model adopts a similar strategy, capitalizing on the
robust prior knowledge embedded in our foundational T2V model to achieve
high-quality I2V generation.

### Model Design

We introduce an additional condition image as the first frame to control
video synthesizing. Specifically, we concatenate the condition image
$I \in {R^{C\times 1\times H\times W}}$ with zero-filled frames along
the temporal axis, and these guidance frames
$I_{c} \in {R^{C\times T\times H\times W}}$ are compressed by -VAE into
condition latent $z_{c} \in {R^{c\times t\times h\times w}}$ ($c=16$
denotes the latent channel, $t=1+(T-1)/4$, $h=H/8$, $w=W/8$).
Additionally, we introduce a binary mask
$M \in \{0,1\}^{1\times T\times h\times w}$ where $1$ indicates the
preserved frame and $0$ denotes the frames to be generated. The spatial
size of mask $M$ is consistent with condition latent $z_{c}$, but $M$
shares the same temporal length with the target video, which is then
rearranged as shape of ${s\times t\times h\times w}$ where $s$ is
temporal stride of -VAE. The noise latent $z_t$, condition latent $z_c$,
and rearranged mask $m$ are concatenated along the channel axis and then
passed through the DiT model of . Since the input of the I2V DiT model
has more channels than T2V ($2\times c + s$ v.s. $c$), an additional
projection layer is employed, which is initialized with zero values.
Moreover, we utilize the image encoder of CLIP  to extract feature
representations from the condition image, and the extracted features are
projected by a three-layer multi-layer perceptron (MLP), which plays a
role as global context. The produced global context is then injected
into the DiT model via decoupled cross-attention.

In practice, the aforementioned approach as illustrated in
Fig. 15 is extensible to other
controllable generation tasks, such as first-last frame transformation
and video continuation. The proposed masking mechanism explicitly
delineates the input conditions and specifies the frames to be
generated. To optimize the efficacy of the masking strategy, we
incorporate multiple tasks, image-to-video generation, video
continuation, first-last frame transformation, and random frame
interpolation, into our unified framework. Specifically, we adopt the
following training paradigm for the I2V model. During the joint training
phase, we conduct unified pre-training of our mask-guided model across
these diverse tasks. This phase facilitates the model’s ability to
discern which positions should be preserved and which should be
generated, depending on the input content. Subsequently, in the
fine-tuning stage, we concentrate on refining the model’s performance on
each individual task.

### Dataset

During the joint training phase, we utilize the same training dataset
employed for T2V pertaining. This phase equips the model with the
foundational capability to generate motion conditioned on reference
images. In the fine-tuning phase, we curate task-specific datasets
tailored to the unique characteristics of each individual task.

**Image-to-video dataset**. In the early experiments, we found that when
the first frame in the training data significantly differed from the
video content, the model struggled to learn stable image-driven video
generation. From this perspective, a smaller difference between the
first frame and the video content in the training data is beneficial for
model training. We calculate the difference between the first frame and
the video content based on SigLIP  features. Specifically, we compute
the cosine similarity between the features of the first frame and the
mean of the features of the remaining frames, and then we retain only
the videos with a similarity greater than a predefined threshold.



**Video continuation dataset.** Similarly, our findings also indicate
that training on temporally consistent videos significantly improves the
performance of video continuation tasks. To quantify the consistency
between the initial and final segments of a video (the first 1.5 seconds
and the last 3.5 seconds), we compute the cosine similarity of their
respective SigLIP features, followed by using the computed similarity
scores to filter and select the required training dataset.

**First-last frame transformation dataset.** Unlike the I2V task, the
community places greater emphasis on the smooth transition between the
initial and final frames in the mage transition task. To this end, we
increase the proportion of data samples with significant transition
between the first and last frames in the training dataset.





### Evaluation

The training of the I2V model, along with first-last frame
transformation and video continuation, is structured into two stages. In
the initial pre-training stage, we employ the same dataset as that used
in the T2V task. Given the slightly larger number of frames involved in
this stage, the image encoder branch is excluded, which is motivated by
the need to maintain a strong alignment between textual input and visual
output, ensuring that the model retains its sensitivity to nuanced
textual semantics. Upon the stabilization of the pre-training stage, the
model has already manifested a variety of capabilities, including I2V
generation, video continuation, and so on. This demonstrates the
efficacy of the pre-training process in establishing a robust foundation
for both spatial and temporal understanding.

Our empirical observations reveal that relying solely on frame-level
information is inadequate for achieving optimal performance in specific
tasks, such as image-to-video generation, video continuation, and
first-last frame transformation.

This insight indicates that while the pre-trained model offers a
foundational framework for frame-based video generation, it lacks the
requisite contextual and semantic depth to fully capture the temporal
consistency of dynamic video sequences, particularly when the number of
conditioning frames is highly limited.

To address this limitation, during the SFT stage, we incorporated an
image encoder to extract image features through decoupled
cross-attention  like , providing global contextual information to
enhance the model’s capacities. We conducted experiments based on two
pre-trained checkpoints at 480p and 720p resolutions, covering the
training of image-to-video generation, video continuation, and
first-last frame transformation tasks.

Similarly to human evaluation for our T2V model, we benchmark our I2V
model against state-of-the-art techniques to provide a comprehensive
comparison focusing on key aspects such as visual quality, motion
quality, and matching. The results shown as
Table 2 demonstrate that our model
performs favorably across all dimensions evaluated.
Fig. 16 and
Fig. 17 present additional
visualizations of our I2V model, offering a clear illustration of its
superior performance in animating images into high-quality videos.



|                 |         |         |         |         |                |
|:----------------|:-------:|:-------:|:-------:|:-------:|:--------------:|
|                 | CN-TopA | CN-TopB | CN-TopC | CN-TopD | **All Rounds** |
| Visual Quality  |  29.2%  |  60.8%  |  24.6%  |  55.6%  |      896       |
| Motion Quality  |  21.7%  |    %    |  32.5%  |  67.0%  |      890       |
| Matching        |  -4.2%  |  35.0%  |  51.7%  |  72.2%  |      896       |
| Overall Ranking |  10.8%  |  47.5%  |  50.8%  |  81.6%  |      890       |

Win rate gap of I2V models. The values in the table represent the
proportion of instances in which the -I2V model was preferred in
pairwise comparisons against other models, relative to the total number
of comparisons conducted.



## Unified Video Editing

The foundational pre-trained models for text-to-image and text-to-video
generation have facilitated the expansion of various downstream tasks
and applications, including repainting , editing , controllable
generation , customized generation , frame reference generation ,
efficient generation , and ID-referenced video synthesis . To enhance
task flexibility and minimize the overhead associated with deploying
multiple models, researchers have increasingly focused on developing
unified model architectures , such as ACE  and OmniGen . These unified
architectures aim to integrate diverse tasks within a single image
model, thereby facilitating the creation of various application
workflows while maintaining ease of use. In the domain of video, the
collaborative transformations in both temporal and spatial dimensions
suggest that leveraging a unified model can unlock limitless
possibilities for video creation.

In our previous work, VACE , we introduced a unified framework for
controllable video generation and editing, addressing tasks such as
reference-to-video generation, video-to-video editing, and masked
video-to-video editing. Building upon the pre-trained model of , we
integrate multiple modalities, including images and videos for editing,
references, and masks, into the Video Condition Unit (VCU) to support
diverse input formats. VACE employs a concept decoupling strategy that
enables the model to identify which aspects should be retained and which
should be modified, thereby distinctly separating visual modality
information in editing and reference tasks. We offer two training modes.
The first mode involves training with the VCU as input, fully
fine-tuning the entire model. The second mode utilizes a pluggable
Context Adapter structure in a Res-Tuning  manner, allowing support for
controllable and editing tasks without modifying the base model’s
weights but with an increased overall model scale. Leveraging the robust
video generation capabilities of , this innovative framework
demonstrates significant competitiveness in both quantitative and
qualitative evaluations. It facilitates the compositional expansion of
fundamental tasks, enabling scenarios such as long video re-rendering.
Consequently, the framework offers a versatile and efficient solution
for video synthesis, opening new avenues for user-driven video content
creation and editing.



### Model Design

The Video Condition Unit (VCU), proposed in VACE, serves as an input
paradigm that unifies diverse input conditions into textual inputs,
frame sequences, and mask sequences. It can be formally represented as
follows: $$V=[T;F;M],$$ where $T$ is a text prompt, while $F$ and $M$
are sequences of context video frames $\{u_1,u_2,...,u_n\}$ and masks
$\{m_1,m_2,...,m_n\}$ respectively. Here, $u$ is in RGB space,
normalized to $[-1, 1]$ and $m$ is binary, in which “1"s and “0"s
symbolize where to edit or not. $F$ and $M$ are aligned both in spatial
size $h \times w$ and temporal size $n$. Under this paradigm, we employ
the previously described -VAE to tokenize the input video frames and
mask information into context tokens. As illustrated in
Fig. 18 (a), these context tokens
are then combined with noisy video tokens to fine-tune the model.
Additionally, a Context Adapter Tuning strategy is proposed, which
allows context tokens to pass through Context Blocks and be reintegrated
into the original DiT blocks. This approach facilitates the seamless
integration of contextual information, enhancing the model’s ability to
handle diverse editing and generation tasks without altering the base
model’s weights.









Before tokenization, pixel-level context frames and masks are
preprocessed using a concept decoupling strategy and then encoded into
the latent space. As shown in
Fig. 18 (b), the concept
decoupling strategy explicitly separates the data of different
modalities and distributions based on masks into two frame sequences of
identical shape: $F_c = F \times M$ and $F_k = F \times (1-M)$, where
$F_c$ refers to reactive frames containing all pixels to be modified,
while $F_k$ denotes inactive frames that retain all pixels to be
preserved, named inactive frames. This mechanism ensures clear task
definitions and guarantees the model’s convergence across different
tasks. $F_c$, $F_k$ are processed by -VAE and mapped into the same
latent space of $X$, maintaining their spatio-temporal consistency. To
avoid any mishmash of images and videos, reference images are separately
encoded by the -VAE encoder and concatenated back along the temporal
dimension, while the corresponding parts need to be removed during
decoding. $M$ is directly reshaped and interpolated. After that, $F_c$,
$F_k$, and $M$ are mapped into latent spaces and spatio-temporal aligned
with $X$ in the shape of $n' \times h' \times w'$.







### Datasets and Implementation

**Data construction**. To develop an all-in-one model, the diversity and
complexity of data construction must increase. For controllable video
generation and editing tasks, input modalities are expanded to include
target videos, source videos, local masks, references, and more.
Efficient and rapid data acquisition for various tasks necessitates
maintaining video quality while performing instance-level analysis and
understanding. We begin by performing shot slicing on the video data and
initially filtering based on resolution, aesthetic score, and motion
amplitude. The first frame is then labeled using RAM  and processed with
Grounding DINO  for detection, enabling secondary filtering of videos
with target areas that are either too small or too large. Additionally,
we utilize the propagation operation of SAM2  for video segmentation to
obtain instance-level information throughout the video. Based on the
segmentation results, we further filter instances temporally by
calculating the effective frame ratio according to a mask area
threshold. The construction for different tasks must be tailored to each
task’s specific characteristics. For detailed information, refer to the
work .

**Implementation details**. We train the controllable and editing model,
pre-trained on -T2V-14B, to support resolutions up to 720p through a
multi-stage training process. Initially, we focus on foundational tasks
such as inpainting and extension to complement the pre-trained
text-to-video models. This phase incorporates masks and develops
contextual generation capabilities in both spatial and temporal
dimensions. Subsequently, we expand the task by transitioning from
single-input to multiple-input reference frames and from individual to
composite tasks. Finally, we enhance the model’s quality by fine-tuning
it with higher-quality data and longer sequences.

### Evaluation

In Fig. 19 and
Fig. [fig:video_editing_showcase2],
we present the results of the single model across various tasks. It is
evident that the model achieves a high level of performance in video
quality and temporal consistency. Additionally, we demonstrate new
applications resulting from the combination of different generative
capabilities, as illustrated in
Fig. [fig:video_editing_showcase3]
(a). Our model performs effectively on these tasks, showcasing
significant potential for capability expansion. The unified controllable
generation and editing framework proposed by VACE is also applicable to
image generation and editing, as shown in
Fig. [fig:video_editing_showcase3]
(b).

## Text-to-Image Generation

is jointly trained on both image and video datasets, equipping it with
not only advanced video generation capabilities but also exceptional
image synthesis performance. This dual-training strategy facilitates
cross-modal knowledge transfer, creating a highly versatile framework
that achieves powerful results in both domains. In practice, our model
has been trained on an image dataset nearly ten times larger than its
video counterpart, allowing for synergistic benefits between image and
video generation tasks. This extensive training regimen has led to
outstanding performance in various aspects of image synthesis. As
demonstrated in Fig. 20, generates high-fidelity images
across diverse categories, encompassing artistic text-based visuals,
photorealistic portraits, imaginative creative designs, and
professional-grade product photography.



## Video Personalization

Video personalization aims to generate videos that maintain consistent
identity with the user-provided reference . In this section, we
integrate advanced personalization techniques into the video generation
pipeline, achieving state-of-the-art performance. The subsequent
discussion provides a comprehensive elaboration of our approach,
encompassing the model architecture, personalization data, and
experimental validation.

### Model Design

The core techniques behind video personalization tackle two primary
challenges: (1) the acquisition of high-fidelity personalized identities
and (2) the seamless integration of these identity features into the
video generation pipeline.

We start from our -T2V foundation model by first obtaining personalized
identity information. Existing video personalization approaches usually
rely on ID extractors (*e.g.,* ArcFace ) or versatile visual extractors
(*e.g.,* CLIP ) to obtain identity information. While they make
promising progress, their performance is bottlenecked by the limitations
of the feature extractors. For instance, ID extractors mainly focus on
features for face recognition but may not capture other critical visual
cues such as scars or stickers. Moreover, they are susceptible to
low-resolution faces, illumination variations, and partial occlusions
(such as glasses, masks, or hair). On the other hand, versatile visual
extractors are not specifically tailored to the face domain and tend to
focus on coarse-grained semantic information. Therefore, we choose not
to rely on any feature extractor to avoid information loss, and our
generation process is directly conditioned on the input face image in
the latent space of -VAE.

There are many ways to inject the personalized identity information into
the video generation process. In our experiments, we found that
cross-attention operation is suited to interact with dense
representations obtained from identity extractors, while self-attention
is more appropriate for modeling data in the same latent space.
Fig. 21 illustrates the
design of our video personalization approach, which follows a
self-attention paradigm. Specifically, in the latent space, we first
extend $K$ additional frames prepended to the given video using the
segmented face images. These face images are extracted from the paired
video using face detection and segmentation, and the facial landmarks
are further aligned to a black canvas that has the same size as the
video frame. Then, along the channel dimension of this extended video,
we concatenate the face images with all-ones masks for the first $K$
frames, and blank images with all-zeros masks for the remaining frames.
These concatenated signals serve as the conditions. Finally, the
diffusion process is performed on this temporally extended video,
conditioned on the channel-wise condition signals in an inpainting
fashion. Note that no other modifications are applied to our -T2V model.

During training, we randomly drop a portion of face images in the $K$
extended frames, so as to support $0$ to $K$ reference face(s) video
generation. In the inference stage, starting from the pure noise of the
extended video, we concatenate the user-provided face images (no more
than $K$) to the extended frames in the channel dimension and set the
corresponding masks as all-ones. The goal is to reconstruct the face
images in the first $K$ frames, and synthesize a new personalized video
in the subsequent frames that preserves the provided identity.



### Dataset

We curate a collection of personalization data from our T2V foundation
model’s training dataset through careful filtering and automatic data
synthesizing. We first filter out about $\mathcal{O}(100)$M videos using
our internal human classifiers and perform face detection at 1 FPS on
these videos. A video shall be discarded if more than one face is
detected in any frame or if over 10% of the frames have no face
detected. Then we calculate ArcFace similarities between consecutive
frames and discard videos with low similarities. Afterwards, we perform
face segmentation to remove the distracting backgrounds, and perform
face landmark detection so as to facilitate canvas alignment during our
training. Note that we do not filter out faces with the small areas,
because we find that such videos usually contain full-body human
figures. Finally, we construct about $\mathcal{O}(10)$M personalized
videos, where each video is accompanied by 5 segmented faces in average.

We also resort to automatic data synthesizing to improve facial
diversity. Particularly, we randomly select $\mathcal{O}(1)$M
personalized videos from above, and adopt Instant-ID  to synthesize
diverse faces for each video. Notably, we build a text template with
over 100 prompts, including anime, line art, cinematic, Minecraft, and
so on. Every time we take a random prompt from this template, as well as
a random human pose estimated from the rest of the videos, to serve as
inputs to Instant-ID. We further measure the ArcFace similarity on the
generated faces and filter out those with lower similarities. In total,
we gather about $\mathcal{O}(1)$M videos with synthesized faces, which
greatly improve the diversity of styles, poses, illuminations, and
occlusions on our personalization dataset.



|                    |        |         |         |         |     |
|:-------------------|:------:|:-------:|:-------:|:-------:|:---:|
|                    |        | CN-TopA | CN-TopB | CN-TopC |     |
| Arcface Similarity | 0.5526 | 0.5655  | 0.5197  | 0.4998  |     |

Comparison of other leading video personalization methods.



### Evaluation

We present our video personalization results in
Fig. 22, where the left part represents the
input identities and the right part represents the synthesized
personalized videos. All the testing images are randomly picked from
Pexels[^1]. We also evaluate our model on an unseen evaluation set and
measure the ArcFace similarity between input faces and output
personalized videos by sampling faces at 1 FPS from videos. The final
similarity scores are averaged from these sampled faces. The
corresponding results are illustrated in
Table 3, where our
approach demonstrates competitive video personalization performance
compared to other leading commercial and closed-source Chinese
competitors.





## Camera Motion Controllability

The camera motion control module is designed to accurately match the
motion and viewpoint of the video by leveraging camera trajectory.
Specifically, we utilize the extrinsic parameters
$[R,t] \in \mathbb{R}^{3 \times 4}$ and intrinsic parameters
$K_{f} \in \mathbb{R}^{3 \times 3}$ for each frame. Our approach
consists of two main components to effectively inject camera motion
conditions: camera pose encoder and camera pose adapter, as illustrated
in Fig. 23.



**Camera pose encoder.** First, for each pixel in each video frame, we
use Plücker coordinates to transform the given extrinsic and intrinsic
parameters into a sequence of fine-grained positions
$P\in \mathbb{R}^{6\times F\times H \times W}$. To adjust the
compression resolution of video latent features, we apply the
PixelUnshuffle operation in Pytorch  to reduce the spatial resolution of
$P$ while increasing the number of channels. Finally, we encode the
output using a series of convolutional modules aligned with the number
of DiT blocks to extract multi-level camera motion features.

**Camera pose adapter.** To integrate camera motion features into video
latent features, we adopt an adaptive normalization layer. Specifically,
we transform the input camera motion feature sequence into scaling
factors $\gamma_{i}$ and shifting parameters $\beta_{i}$ using two
zero-initialized convolution layers. Then, $\gamma_{i}$ and $\beta_{i}$
are incorporated into each DiT block through a simple linear projection
defined by the formula $f_{i} =(\gamma_{i}+1)*f_{i-1}+ \beta_{i}$, where
$f_{i}$ represents the video latent features at layer $i$.

Regarding the training data, we utilize an advanced camera pose
estimation algorithm named VGG-SfM   to extract camera trajectories from
our training videos that exhibit significant camera motion. This process
yields approximately $\mathcal{O}(1)$ thousand video segments. We train
our module within our text-to-video generation framework using the Adam
optimizer.
Fig. 24 presents various
example results of videos generated under camera motion guidance.

## Real-time Video Generation

Current video generative methods often require significant computational
resources and considerable time to produce even short video clips.
Systems relying on high-end hardware remain slow, often taking several
minutes to generate just a few seconds of video. While the visual
quality of these generated videos has seen substantial improvements in
recent years, the extended processing times present a major bottleneck
in practical applications and iterative design workflows. In fields
where rapid prototyping and real-time adjustments are crucial—such as
interactive entertainment, virtual reality, and video production—these
long generation times limit creative flexibility and slow down the
process of refining visual content.

Moreover, the slow speed of current methods limits their use in dynamic
environments where real-time feedback is essential. In contexts like
live-streaming or gaming, there’s an increasing demand for systems that
can instantly render synthetic scenes and characters in response to user
actions or changing inputs. Any delay in generating high-quality video
clips can result in a suboptimal user experience, reducing immersion and
interactivity. By focusing on real-time generation, seeks to enhance
both the performance and applicability of video generation techniques,
broadening their potential in fast-paced, interactive environments.
Fig. [fig:streamer1] and
[fig:streamer2] provide such cases of
real-time long video stream generation. We then introduce how to achieve
this using our models.













### Method

To construct the real-time generation pipeline, we build upon a
previously pre-trained version of the model. This design choice offers
two primary benefits. First, starting from a well-trained model
significantly accelerates convergence and improves training stability,
as the model already possesses a strong initialization. Second, the
pre-trained model has already captured valuable knowledge about motion
patterns and continuous temporal dynamics, which can be directly
inherited by the new pipeline to enhance the smoothness and coherence of
generated content. Therefore, we adopt the pre-trained model as the
foundation for our real-time pipeline [^2].

The pre-trained model is designed to generate fixed-length video clips,
typically 5 to 10 seconds long. To transform this into a real-time
streaming model capable of generating continuous video streams with no
predefined length (potentially infinite), two key modifications are
required:

- **Streaming pipeline adaptation.** We replace the static generation
  process, which produces a fixed-length sequence of video tokens in a
  single pass, with a streaming mechanism that incrementally generates
  video tokens. In this streaming setup, video tokens are processed
  through a denoising queue — each time the oldest token is generated
  and dequeued, a new token is added to the bottom of the queue,
  enabling continuous generation without length constraints.

- **Real-time acceleration.** In addition to enabling streaming, we
  optimize the generation speed to ensure the system meets real-time
  performance requirements, the pipeline must generate new frames fast
  enough to keep up with real-time playback.

In the following sections, we will describe each of these modifications
in detail, including their design rationale, technical implementation,
and empirical performance.

### Streaming Video Generation

Traditional Diffusion Transformer (DiT)  models are constrained in their
ability to generate videos longer than a few seconds, even when
employing significant spatial and temporal compression through
Variational Autoencoders (VAEs) . This limitation stems primarily from
the computational and memory-intensive nature of attention mechanisms
over extended temporal sequences. To overcome this challenge, we
introduce Streamer, a novel approach that leverages a sliding temporal
window to manage temporal dependencies efficiently, enabling the
generation of long or even infinite-length videos in a streaming
fashion.

#### Shift window denoise process models.

The key innovation of Streamer lies in its assumption that temporal
dependencies are confined within a limited time window. By focusing
attention computations within this window, Streamer significantly
reduces computational overhead while maintaining video continuity.
Specifically, Streamer processes video tokens in a queue, denoising them
over multiple noise levels within the window. After a fixed number of
denoising steps, the leftmost token (with the lowest noise level) is
dequeued and cached, while a new token with Gaussian noise is appended
to the rightmost position. This sliding mechanism ensures that the
window maintains a consistent length while facilitating the generation
of continuous video frames.

#### Training and inference.

Streamer is fine-tuned from a pre-trained DiT model. During training, a
sequence of $2w$ video tokens is sampled, where $w$ is the size of the
sliding window (typically set to $w=T$, the number of diffusion solver
steps). The first $w$ tokens are used solely for ‘warming up’ the model
and do not contribute to the loss computation. The loss is calculated
only on the last $w$ tokens, ensuring that the model learns to generate
coherent video frames within the window. At inference time, the same
warmup strategy is employed: the first $w$ tokens are discarded, and the
generated video begins from the $(w+1)$-th token.

#### Ensuring continuity during training and inference.

To maintain smooth transitions between consecutive windows, cached
tokens are re-introduced into the token queue at a noise level of 0.
This allows previously generated tokens to participate in the denoising
process of subsequent windows, ensuring temporal consistency and
continuity in the generated video.

Streamer represents a significant advancement in video generation by
adapting DiT models for streaming applications. Its sliding temporal
window approach addresses the limitations of traditional DiT models,
enabling the efficient generation of infinite-length videos while
maintaining high quality and temporal coherence. This method opens new
possibilities for real-time video generation and long-form content
creation. We summarize its advantages as follows:

- Infinite-Length Video Generation: By leveraging a sliding window
  mechanism, Streamer can generate videos of arbitrary length without
  incurring prohibitive computational costs;

- Efficient Attention Computation: Limiting attention computations to a
  confined temporal window reduces memory and processing demands;

- Seamless Continuity: The caching and re-introduction of tokens ensure
  smooth transitions between windows, preserving video quality over
  extended durations.

### Consistency Model Distillation

After successfully extending the DiT model to Streamer for
infinite-length video generation, the next critical step is to address
the challenge of achieving real-time rendering of the simulated world.
While Streamer enables the generation of long or even infinite videos by
efficiently managing temporal dependencies within a sliding window, the
computational demands of the diffusion process still pose a bottleneck
for real-time applications. To overcome this, we propose a novel
integration of Streamer with Consistency Models, a cutting-edge approach
for accelerating diffusion-based generative processes. Specifically, we
leverage the Latent Consistency Model (LCM ) and its video version
(VideoLCM ), which distill the original diffusion process and class-free
guidance into a highly efficient four-step consistency model. This
distillation process significantly reduces the number of sampling steps
required for high-quality generation while maintaining the temporal
coherence and streaming capabilities inherent to Streamer.

The integration of LCM with Streamer is designed to preserve the
denoising window mechanism, ensuring that the sliding temporal window
continues to manage dependencies effectively. During training, we
optimize this combined framework to balance efficiency and quality. The
result is a 10 - 20 $\times$ acceleration in inference speed, enabling
the model to achieve a rendering rate of 8 - 16 FPS. This dramatic
improvement in performance makes the system viable for real-time
applications, such as interactive simulations, live video synthesis, and
dynamic content generation in virtual environments. By combining the
strengths of Streamer—its ability to generate infinite-length videos
with temporal coherence—and LCM—its efficiency in accelerating the
diffusion process—we bridge the gap between high-quality video
generation and real-time responsiveness. This advancement not only
enhances the practicality of diffusion models for streaming applications
but also opens up new possibilities for immersive, interactive, and
real-time media creation.

#### Quantization for running in customer-level devices.

While the generating speed has been accelerated to real-time levels,
deploying the model on consumer-level devices remains challenging due to
the high computational and memory requirements, even for high-end GPUs
like the NVIDIA 4090. To address this, we introduce quantization
techniques to optimize the model for efficient deployment. Specifically,
we employ two distinct quantization strategies: int8 quantization  for
attention layers and linear heads, and TensorRT quantization  for the
overall model. The int8 quantization approach significantly reduces
memory consumption by converting weights and activations to 8-bit
integers, while maintaining the overall generation quality. However,
this method provides minimal acceleration in generating speed, as it
primarily focuses on memory efficiency rather than computational
optimization. On the other hand, TensorRT quantization offers a more
comprehensive solution, enabling substantial acceleration in generation
speed. This technique allows the model to achieve real-time performance
of 8 FPS even on a single 4090 GPU, making it feasible for
consumer-level devices. However, TensorRT quantization introduces a
modest increase in network error, which can be observed even when its
built-in error-checking mechanisms are enabled. This error manifests as
slight deviations in output quality, such as minor artifacts or
inconsistencies in the generated video. In some cases, it may also lead
to measurable instability, such as flickering or temporal incoherence.
Despite these trade-offs, the combination of int8 and TensorRT
quantization provides a balanced approach to optimizing the model for
real-time, consumer-level deployment, ensuring both efficiency and
practicality while maintaining acceptable generation quality. By
carefully tuning the quantization parameters and leveraging TensorRT’s
error-checking features, we mitigate these issues to the greatest extent
possible, enabling robust and efficient performance on consumer-grade
hardware.

## Audio Generation

The primary objective of audio generation in this work is to produce
synchronized soundtracks for video clips, formulated as a video-to-audio
(V2A) generation framework. The generated soundtracks consist of ambient
sound and background music, explicitly excluding speech or vocal
elements. In contrast to text-to-audio models  , the ambient sound
generated by video-to-audio models must be temporally aligned with the
visual content of the video, while the accompanying music should
accurately reflect the emotional tone and contextual setting of the
video, as would naturally be expected. Furthermore, to enhance user
control over sound design, our framework integrates both video and
textual prompts, enabling users to specify on-screen or off-screen
sounds and define the style and presence of background music.

### Model Design

Consistent with our video generation framework, our video-to-audio model
also utilizes a diffusion transformer (DiT)   based on the
flow-matching  to model the denoising diffusion process in the audio
domain. The detailed pipeline of our V2A model is illustrated in
Fig. 25.





**Audio autoencoder**. A widely adopted audio compression approach
involves transforming raw waveforms into mel-spectrograms, followed by
encoding them into $H^{a}\times W^{a}\times C^{a}$ latent features using
an image-based variational autoencoder, as demonstrated in  . However,
in the context of a diffusion transformer backbone, these latent
features must be partitioned into patches and reshaped into a tensor of
size $(H^a\cdot W^a)\times C^a$. This patching-and-reshaping process can
disrupt the temporal alignment with the corresponding video content,
posing a significant challenge for synchronization. Considering this
limitation, we train a compression model that operates directly on raw
waveforms. Our 1D-VAE generates latent features of size
$T^{a}\times C^{a}$, where $T^a$ represents the sequence length along
the time axis, preserving explicit temporal information essential for
accurate synchronization.

**Video and text encoder**. To achieve frame-accurate synchronization
between synthesized audio and visual sequences, we establish temporal
coherence through multimodal feature fusion. Our architecture first
extracts frame-wise visual embeddings using a CLIP  model, then performs
temporal rate adaptation through feature replication to match the audio
feature’s sampling rate, following . The synchronized visual and
acoustic features undergo dimension transformation via linear projection
layers to align with DiT’s latent space, followed by element-wise
summation for multimodal fusion. For linguistic understanding across
languages, we leverage the pre-trained umT5 model  as our text encoder,
capitalizing on its cross-lingual representation capabilities through
frozen parameters.

**Data.** The training data for our V2A model is derived from the video
generation dataset through a rigorous filtering process. We
systematically remove videos lacking soundtracks or containing
speech/vocal music, resulting in a refined subset of $\mathcal{O}(1)$
thousand hours. To enhance multimodal representation, we employ a
comprehensive captioning strategy: dense video descriptions are
complemented with audio-specific captions generated by Qwen2-audio .
These audio captions are categorized into ambient sounds and musical
compositions, with the latter characterized by style, rhythm, melody,
and instrumentation. The final structured caption integrates three
components: (1) dense video description, (2) ambient sound
characterization, and (3) background music analysis, providing a unified
multimodal representation for training.

**Implementation details.** Our V2A architecture generates high-fidelity
stereo audio with a maximum duration of 12 seconds at a 44.1 kHz
sampling rate. The model processes multimodal inputs through dedicated
encoders: umT5-XXL produces 4096-dimensional text embeddings, while the
CLIP extracts 1024-dimensional visual embeddings per frame. The audio
and video features are projected into a unified 1536-dimensional latent
space within the DiT backbone. For temporal alignment, we downsample
input videos and yield precisely 48 frames for 12-second clips, ensuring
frame-accurate synchronization between visual and audio modalities. The
system processes latent sequences of length 256 in one batch. To enhance
the model’s capability of generating audio solely from visual cues, we
implement a random masking strategy during training where ambient sound
and music captions are selectively omitted with a predefined
probability. This conditioning mechanism forces the model to establish
robust cross-modal associations between visual content and corresponding
audio patterns, while maintaining the flexibility to utilize textual
descriptions when available.

### Evaluation

We provide qualitative evaluations of audio generation in
Fig. 26, where we compare our
method with the recently released open-sourced approach MMAudio . This
comparative analysis is particularly meaningful as MMAudio represents
the most similar and competitive baseline to our method, demonstrating
impressive performance in audio generation tasks. The videos are
generated by our text-to-video model with a duration of 5 seconds.

Our V2A model demonstrates superior performance in several key aspects
of audio generation, as evidenced by the comparative results.
Specifically, the model exhibits enhanced long-term consistency,
particularly noticeable in the ‘pouring coffee’ scenario (the first case
in Fig. 26). In the second case
of ‘typing’, our approach generates significantly cleaner audio output
compared to the noisier results produced by MMAudio. Furthermore, our
method excels in synthesizing rhythmic sound patterns, as demonstrated
in the ‘walking’, ‘boxing’, and ‘clip-clop’ examples (the last three
cases in
Fig. 26), where it achieves
more natural and coherent audio generation.

**Limitations.** Our method exhibits limitations in generating human
vocal sounds, including but not limited to laughter, crying, and speech.
This limitation is primarily attributed to our data preparation process,
where speech-related vocal data is deliberately excluded from the
training dataset. The MMAudio model’s ability to produce random
speech-like sounds stems from its retention of speech-related data in
its training corpus. In our future work, we plan to incorporate speech
generation capabilities to address this limitation.

# Limitation and Conclusion

**Limitation**. This study presents , a foundational video generation
model that has achieved notable advancements across multiple benchmarks.
Specifically, significant improvements have been observed in motion
amplitude (*e.g*., sports, dance) and instruction-following
capabilities. Nevertheless, several limitations persist, which should be
addressed to realize more potential of . First, preserving fine-grained
details in scenarios involving large motion remains a challenge for our
method, as well as a broader issue within the field of video generation.
Substantial efforts are required to improve the fidelity of videos with
large-scale motion dynamics. Second, the computational cost associated
with large-scale models remains prohibitive. Inference on a 14B model
currently requires approximately 30 minutes on a single head-end GPU
without additional optimization. To democratize video generation as a
universally accessible AI tool, further research into efficiency and
scalability is essential. Finally, there is still a lack of
domain-specific expertise. As a foundational video model, we are
committed to enhancing ’s general capabilities, yet performance in
specific localized scenarios, such as education and medicine, may be
insufficient. We aim to address this limitation by open-sourcing our
latest models and fostering community-driven development across diverse
specialized domains.

**Conclusion**. In this report, we publicly release , our latest video
model, which establishes a new benchmark for video generation. We
provide a comprehensive overview of the architectural design of our -VAE
and DiT models, including detailed insights into their training
approach, data curation processes, evaluation methods, and empirical
results. Furthermore, we meticulously analyze our data preprocessing
pipeline and the strategic adaptations implemented to optimize model
training. This holistic approach is designed to drive advancements in
the domain of video generation. We also extensively investigate
downstream applications, such as image-to-video generation, video
editing, and personalized video generation, to demonstrate the
versatility and practical utility of across diverse scenarios. In
addition to open-sourcing the 14B model, we explore the feasibility of
leveraging smaller-scale models for efficient video generation. Notably,
our 1.3B model not only achieves competitive performance compared to
larger counterparts but also enables seamless inference on
consumer-grade GPUs, significantly enhancing accessibility and
practicality for content creators. Looking ahead, we plan to focus on
scaling both data and model architectures to tackle the most pressing
challenges in video generation. Our ongoing efforts aim to provide the
research community with more robust and versatile tools for video
creation, fostering innovation and broader adoption in this rapidly
evolving field.

# Contributors

Authors are listed **alphabetically by the first name**.



3 Ang Wang  
Baole Ai  
Bin Wen  
Chaojie Mao  
Chen-Wei Xie  
Di Chen  
Feiwu Yu  
Haiming Zhao  
Jianxiao Yang  
Jianyuan Zeng  
Jiayu Wang  
Jingfeng Zhang  
Jingren Zhou  
Jinkai Wang  
Jixuan Chen  
Kai Zhu  
Kang Zhao  
Keyu Yan  
Lianghua Huang  
Mengyang Feng  
Ningyi Zhang  
Pandeng Li  
Pingyu Wu  
Ruihang Chu  
Ruili Feng  
Shiwei Zhang  
Siyang Sun  
Tao Fang  
Tianxing Wang  
Tianyi Gui  
Tingyu Weng  
Tong Shen  
Wei Lin  
Wei Wang~1  
Wei Wang~2  
Wenmeng Zhou  
Wente Wang  
Wenting Shen  
Wenyuan Yu  
Xianzhong Shi  
Xiaoming Huang  
  
Xin Xu  
Yan Kou  
Yangyu Lv  
Yifei Li  
Yijing Liu  
Yiming Wang  
Yingya Zhang  
Yitong Huang  
Yong Li  
You Wu  
Yu Liu  
Yulin Pan  
Yun Zheng  
Yuntao Hong  
Yupeng Shi  
Yutong Feng  
Zeyinzi Jiang  
Zhen Han  
Zhi-Fan Wu  
Ziyu Liu



[^1]: www.pexels.com

[^2]: Part of this section is introduced in a previously released tech
    report *The Matrix* .