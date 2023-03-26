# %%
import collections
import os
from os.path import join
import sys
import random
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from random import sample
from GAT import *

DATA_PATH = '../data/SL/'
KG_FILE_NAME = 'raw/kg2id.txt'
SL_DATASET = 'raw/sl2id.txt'
NEIGHBOR_NUM = 16


# %%
class KGDataset(Dataset):
    def __init__(self, kg_path=join(DATA_PATH, KG_FILE_NAME)):
        kg_data = pd.read_csv(kg_path,
                              sep='\t',
                              names=['h', 'r', 't'],
                              engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads, self.rels, self.tails = self.generate_kg_data(
            kg_data=self.kg_data)
        self.item_net_path = join(DATA_PATH, SL_DATASET)
        self.graph = None
        self.sampled_edges, self.sampled_rels = self.get_Neighborhood()

    @property
    def entity_count(self):
        return max(self.kg_data['t'].max(), self.kg_data['h'].max()) + 2

    @property
    def relation_count(self):
        return self.kg_data['r'].max() + 2

    def generate_kg_data(self, kg_data):
        print('generating kg data...')
        try:
            kg_dict = np.load(DATA_PATH + 'kg/kg_dict.npy',
                              allow_pickle=True).item()
            hs = np.load(DATA_PATH + 'kg/hs.npy', allow_pickle=True)
            rs = np.load(DATA_PATH + 'kg/rs.npy', allow_pickle=True)
            ts = np.load(DATA_PATH + 'kg/ts.npy', allow_pickle=True)
            print('successfully loaded...')
        except:
            print('start to generate...')
            s = time()
            #construct kg dict
            hs = []
            rs = []
            ts = []
            kg_dict = collections.defaultdict(list)
            for row in kg_data.iterrows():
                h, r, t = row[1]
                kg_dict[h].append((r, t))
                hs.extend([h])
                rs.extend([r])
                ts.extend([t])
            hs = np.array(hs)
            rs = np.array(rs)
            ts = np.array(ts)
            end = time()
            print(f"cost {end-s}s, saving npy file...")
            np.save(DATA_PATH + 'kg/kg_dict.npy', kg_dict)
            np.save(DATA_PATH + 'kg/hs.npy', hs)
            np.save(DATA_PATH + 'kg/rs.npy', rs)
            np.save(DATA_PATH + 'kg/ts.npy', ts)

        return kg_dict, hs, rs, ts

    def __len__(self):
        return len(self.kg_dict)

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail

    def get_Neighborhood(self):
        print('sampling neighborhood...')
        neighbor_num = NEIGHBOR_NUM
        try:
            edges = np.load(DATA_PATH + 'kg/sample_edges.npy',
                            allow_pickle=True).item()
            rels = np.load(DATA_PATH + 'kg/sample_rels.npy',
                           allow_pickle=True).item()
            print('successfully loaded...')
        except:
            print('start to sample...')
            s = time()
            edges = dict()
            rels = dict()
            for entity in range(self.entity_count):
                rts = self.kg_dict.get(entity, False)
                if rts:
                    tails = np.array(list(map(lambda x: x[1], rts)))
                    relations = np.array(list(map(lambda x: x[0], rts)))
                    if (len(tails) > neighbor_num):
                        random_idx = np.random.choice(range(len(tails)),
                                                      neighbor_num,
                                                      replace=False)
                        edges[entity] = torch.tensor(tails[random_idx])
                        rels[entity] = torch.tensor(relations[random_idx])
                    else:
                        tails = np.append(tails, [self.entity_count - 1] *
                                          (neighbor_num - len(tails)))
                        relations = np.append(relations,
                                              [self.relation_count - 1] *
                                              (neighbor_num - len(relations)))

                        edges[entity] = torch.tensor(tails)
                        rels[entity] = torch.tensor(relations)
                else:
                    edges[entity] = torch.tensor(
                        np.array([self.entity_count - 1] * neighbor_num))
                    rels[entity] = torch.tensor(
                        np.array([self.relation_count - 1] * neighbor_num))
            end = time()
            print(f"cost {end-s}s, saving npy file...")
            np.save(DATA_PATH + 'kg/sample_edges.npy', edges)
            np.save(DATA_PATH + 'kg/sample_rels.npy', rels)

        return edges, rels


# %%
class SLDataset(Dataset):
    def __init__(self, fold_n) -> None:
        super().__init__()
        print(f'loading sl dataset...')
        self.fold_n = fold_n
        self.df_dataset = self.read_csv(path=(DATA_PATH + SL_DATASET))
        self.reindex_dict = self.get_reindex_dict()
        self.an_reindex_dict = {y: x for x, y in self.reindex_dict.items()}
        self.train_df, self.val_df, self.test_df = self.get_df()
        self.train_a, self.train_b, self.train_label, self.val_a, self.val_b, self.val_label, self.test_a, self.test_b, self.test_label = self.get_data(
        )
        self.SLGraph = self.getSLGraph()
        print('done...')

    @property
    def n_gene(self):
        return max(self.reindex_dict.values()) + 1

    @property
    def trainDataSize(self):
        return len(self.train_a)

    @property
    def valDataSize(self):
        return len(self.val_a)

    @property
    def testDataSize(self):
        return len(self.test_a)

    def read_csv(self, path):
        df = pd.read_csv(path, sep='\t')
        df = df[df['label'] == 1][['gene_a', 'gene_b']]
        df_reverse = pd.DataFrame()
        df_reverse['gene_a'] = df['gene_b']
        df_reverse['gene_b'] = df['gene_a']
        df_final = pd.concat([df, df_reverse])
        return df_final

    def get_df(self):
        data_path = DATA_PATH + 'fold_data/fold_' + str(self.fold_n) + '/'
        train_df = pd.read_csv(data_path + 'train.txt', sep='\t')
        val_df = pd.read_csv(data_path + 'val.txt', sep='\t')
        test_df = pd.read_csv(data_path + 'test.txt', sep='\t')
        train_df = self.reindex_sl(train_df)
        val_df = self.reindex_sl(val_df)
        test_df = self.reindex_sl(test_df)

        return train_df, val_df, test_df

    def get_data(self):
        train_a = np.array(self.train_df['gene_a'])
        train_b = np.array(self.train_df['gene_b'])
        train_label = np.array(self.train_df['label'])
        val_a = np.array(self.val_df['gene_a'])
        val_b = np.array(self.val_df['gene_b'])
        val_label = np.array(self.val_df['label'])
        test_a = np.array(self.test_df['gene_a'])
        test_b = np.array(self.test_df['gene_b'])
        test_label = np.array(self.test_df['label'])

        return train_a, train_b, train_label, val_a, val_b, val_label, test_a, test_b, test_label

    def getSLGraph(self):
        gene_a = np.array(self.train_df[self.train_df['label'] == 1]['gene_a'])
        gene_b = np.array(self.train_df[self.train_df['label'] == 1]['gene_b'])
        SLGraph = csr_matrix((np.ones(len(gene_a)), (gene_a, gene_b)),
                             shape=(self.n_gene, self.n_gene))
        SLGraph += sp.eye(SLGraph.shape[0], format='csr')

        return SLGraph

    def getPosNeighbors(self, ids):
        posNeighbors = []
        for id in ids:
            posNeighbors.append(self.SLGraph[id].nonzero()[1])
        return posNeighbors

    def get_reindex_dict(self):
        df = self.df_dataset
        set_a = set(list(df['gene_a']))
        set_b = set(list(df['gene_b']))
        set_all = sorted(list(set_a | set_b))
        reindex_dict = dict()
        for i in range(len(set_all)):
            reindex_dict[set_all[i]] = i
        return reindex_dict

    def reindex_sl(self, df_for_reidx):
        reindex_dict = self.reindex_dict
        df = df_for_reidx.copy()
        df['gene_a'] = df['gene_a'].apply(lambda x: reindex_dict.get(x))
        df['gene_b'] = df['gene_b'].apply(lambda x: reindex_dict.get(x))
        return df
