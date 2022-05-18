import torch;

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from tqdm import tqdm
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from lib.models.vq_utils import vq, vq_st

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VQEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VQEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        return mu

class VQDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(VQDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        latents = vq(z_e_x, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x_):
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())

        z_q_x_bar_flatten = torch.index_select(
                self.embedding.weight,
                dim=0,
                index=indices
        )
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)

        return z_q_x_, z_q_x_bar_