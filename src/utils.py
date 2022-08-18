import argparse
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import dgl
import torch.optim as optim
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from torch._C import device
from tqdm import tqdm

device = 'cuda:0'


def DistanceCorrelation(tensor_1, tensor_2):
    # tensor_1, tensor_2: [channel]
    # ref: https://en.wikipedia.org/wiki/Distance_correlation
    channel = tensor_1.shape[0]
    zeros = torch.zeros(channel, channel).to(tensor_1.device)
    zero = torch.zeros(1).to(tensor_1.device)
    tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
    """cul distance matrix"""
    a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
           torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
    tensor_1_square, tensor_2_square = tensor_1**2, tensor_2**2
    a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
           torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
    """cul distance correlation"""
    A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1,
                                                 keepdim=True) + a.mean()
    B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1,
                                                 keepdim=True) + b.mean()
    dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel**2, zero) + 1e-8)
    dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel**2, zero) + 1e-8)
    dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel**2, zero) + 1e-8)
    return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)




def load_data(args):
    kg2id_np = np.loadtxt('../data/SL/raw/kg2id' + '.txt', dtype=np.int64)
    max = 0
    r_max = 0
    for i in range(len(kg2id_np)):
        h = kg2id_np[i][0]
        t = kg2id_np[i][2]
        r = kg2id_np[i][1]
        if h > max:
            max = h
        if t > max:
            max = t
        if r > r_max:
            r_max = r
    n_entities = max + 1
    n_relations = r_max + 1

    graph = nx.MultiDiGraph()
    for i in range(len(kg2id_np)):
        h = kg2id_np[i][0]
        r = kg2id_np[i][1]
        t = kg2id_np[i][2]
        graph.add_edge(h, t, key=r)
        if args.inverse_r == True:
            graph.add_edge(t, h, key=r + n_relations)
    return graph,n_entities, n_relations


class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        gene_a = np.array(self.df.iloc[idx]['gene_a'])
        gene_b = np.array(self.df.iloc[idx]['gene_b'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return gene_a, gene_b, label


