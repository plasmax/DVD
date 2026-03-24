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
    def __init__(self):
        super(DiffusionTransformer, self).__init__()
        self.timestep = # TODO: can this be a fixed value?
        self.sinusoidal_embedding = sinusoidal_embedding_1d(self.freq_dim, self.timestep)

    def time_embedding(self, t):
        return t

    def forward(self, x):
        return x


class DeterministicVideoDepth(nn.Module):
    # Adapted from WanVideoPipeline
    def __init__(self):
        super(DeterministicVideoDepth, self).__init__()
        self.vae = VariationalAutoencoder()
        self.dit = DiffusionTransformer()
        # Q: does it need the scheduler?

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