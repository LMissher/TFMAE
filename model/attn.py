import torch
import torch.nn as nn
import math
from math import sqrt

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()

        self.norm = nn.LayerNorm(d_model)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # [B, T, D]
        B, T, D = x.shape

        queries = self.query_projection(x)
        keys = self.key_projection(x).transpose(1,2)
        values = self.value_projection(x)

        attn = torch.softmax(torch.matmul(queries, keys) / math.sqrt(D), -1)

        out = torch.matmul(attn, values) + x

        return self.out_projection(self.norm(out)) + out, attn
