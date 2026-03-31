# a one-file bare-bones handwritten refactor for my own education.

import torch
import torch.nn as nn


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000,
                  -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype) 


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (
            torch.arange(0, dim, 2)[: (dim //2)].double() / dim
        )
    )
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex64
    return freqs_cis


def precompute_freqs_cis_3d(dim: int):
    # 3d rotary positional embedding precomputation
    end = 1024
    theta = 10000.0
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


class VariationalAutoencoder(nn.Module):
    # Adapted from WanVideoVAE
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()

    def encode(self, x):
        return x

    def decode(self, x):
        return x


class DiffusionTransformer(nn.Module):
    # Adapted from WanModel
    def __init__(
        self,
        has_image_input: bool = False,
        patch_size: list = [1, 2, 2],
        in_dim: int = 16,
        dim: int = 1536,
        ffn_dim: int = 8960,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 12,
        num_layers: int = 30,
        eps: float = 1e-6,
    ):
        super(DiffusionTransformer, self).__init__()
        self.freq_dim = 256
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        self.timestep = torch.tensor([500.0])
        self.sinusoidal_embedding = sinusoidal_embedding_1d(
            self.freq_dim, self.timestep
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )
        t = self.time_embedding(self.sinusoidal_embedding)
        self.t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # Context is just a zero-initialized tensor with a batch size of 1
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.context = self.text_embedding(
            torch.zeros([1, 512, text_dim])
        )

        # I'm fairly certain this is only needed during training - 
        # we want to make sure the timestep and text embedding dimensions have the same batch size.
        if self.timestep.shape[0] != self.context.shape[0]:
            self.timestep = torch.concat(
                [self.timestep] * self.context.shape[0], dim=0
            )

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def forward(self, x):
        x, (f, h, w) = self.patchify(x)
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        

        return x


class DeterministicVideoDepth(nn.Module):
    # Adapted from WanVideoPipeline
    def __init__(self):
        super(DeterministicVideoDepth, self).__init__()
        self.vae = VariationalAutoencoder()
        self.dit = DiffusionTransformer()

    @classmethod
    def from_pretrained(cls):
        pipe = cls()
        # TODO: load the weights here 
        return pipe

    def forward(self, x):

        latents = self.vae.encode(x)
        x = self.dit(latents)
        x = self.vae.decode(x)

        return x