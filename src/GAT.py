import random
from code import interact
from traceback import print_tb
from turtle import forward

import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
from torch_scatter import scatter_mean

from utils import *

device = 'cuda:0'

class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, num_classes, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(in_dim, num_hidden, heads[0], feat_drop, attn_drop,
                    negative_slope, True, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(num_hidden * heads[l - 1], num_hidden, heads[l],
                        feat_drop, attn_drop, negative_slope, residual,
                        self.activation))
        # output projection
        self.gat_layers.append(
            GATConv(num_hidden * heads[-2], num_classes, heads[-1], feat_drop,
                    attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits
