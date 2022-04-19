import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch._C import device
from tqdm import tqdm

from model import *
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=30, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--dim', type=int, default=64, help='embedding size')
parser.add_argument('--l2',
                    type=float,
                    default=1e-4,
                    help='l2 regularization weight')
parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
parser.add_argument('--sim_regularity',
                    type=float,
                    default=1e-3,
                    help='regularization weight for latent factor')
parser.add_argument("--inverse_r",
                    type=bool,
                    default=True,
                    help="consider inverse relation or not")
parser.add_argument("--node_dropout",
                    type=bool,
                    default=True,
                    help="consider node dropout or not")
parser.add_argument("--node_dropout_rate",
                    type=float,
                    default=0.4,
                    help="ratio of node dropout")
parser.add_argument("--mess_dropout",
                    type=bool,
                    default=True,
                    help="consider message dropout or not")
parser.add_argument("--mess_dropout_rate",
                    type=float,
                    default=0.2,
                    help="ratio of node dropout")

parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
parser.add_argument("--gpu_id", type=int, default=1, help="gpu id")

parser.add_argument("--n_factors",
                    type=int,
                    default=4,
                    help="number of latent factor for user favour")
parser.add_argument("--ind",
                    type=str,
                    default='distance',
                    help="Independence modeling: mi, distance, cosine")

# ===== relation context ===== #
parser.add_argument('--context_hops',
                    type=int,
                    default=3,
                    help='number of context hops')

args = parser.parse_known_args()[0]

graph, sl_adj, n_entities, n_relations = load_data(args)
print('Hyperparameters:')
for key in args.__dict__:
    if key == 'epoch' or key == 'batch_size' or key == 'node_dropout' or key == 'mess_dropout' or key == 'cuda' or key == 'gpu_id':
        continue
    else:
        print('{}: {}'.format(key, args.__dict__[key]))
if (args.inverse_r == True):
    n_relations = n_relations * 2
print('**************************************************')
print('Data information:')
print('n_relation: {}\nn_entities: {}\nn_edges: {}'.format(
    n_relations, n_entities, len(graph.edges)))
print('**************************************************')
print('Start training:')

df_dataset = pd.read_csv('../data/SL/sl2id.txt', sep='\t')
list_all = split_full_G(df_dataset)

list_train = []
list_test = []
for i in range(5):
    list_test.append(list_all[i])
    list_temp = list()
    for index in range(5):
        if index == i:
            continue
        else:
            list_temp.extend(list_all[index])
    list_train.append(list_temp)


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


for i in range(5):
    print('5-fold-validation:{}'.format(i + 1))

    model = SLModel(n_entities, n_relations, n_entities, args, graph,
                    sl_adj).to('cuda:1')

    train_index = list_train[i]
    test_index = list_test[i]
    valid_index, test_index = get_test_val_G(test_index)

    data_train = df_dataset.iloc[train_index]
    data_valid = df_dataset.iloc[valid_index]
    data_test = df_dataset.iloc[test_index]

    train_dataset = KGCNDataset(data_train)
    valid_dataset = KGCNDataset(data_valid)
    test_dataset = KGCNDataset(data_test)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cnt_wait = 0
    best_valid_auc = 0
    best_valid_aupr = 0
    best_valid_f1 = 0
    best_test_auc = 0
    best_test_aupr = 0
    best_test_f1 = 0
    with_loss = 0
    with_cor = 0
    for epoch in range(args.epoch):
        labels_train = torch.tensor([]).to('cuda:1')
        logits_train = torch.tensor([]).to('cuda:1')
        running_loss = 0
        train_cor = 0
        for i, (gene_a, gene_b, labels) in enumerate(train_loader):
            gene_a = gene_a.to('cuda:1')
            gene_b = gene_b.to('cuda:1')
            labels = labels.to('cuda:1')
            logits, emb_loss, cor_loss, cor = model(gene_a, gene_b)
            labels_train = torch.cat((labels_train, labels))
            logits_train = torch.cat((logits_train, logits))
            loss = criterion(logits, labels)
            loss = loss + emb_loss + cor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_cor += cor.item()
        running_loss = running_loss / len(train_loader)
        train_cor = train_cor / len(train_loader)

        with torch.no_grad():
            valid_loss = 0
            total_roc = 0
            valid_cor = 0
            labels_valid = torch.tensor([]).to('cuda:1')
            logits_valid = torch.tensor([]).to('cuda:1')
            for i, (gene_a, gene_b, labels) in enumerate(valid_loader):
                gene_a, gene_b, labels = gene_a.to('cuda:1'), gene_b.to(
                    'cuda:1'), labels.to('cuda:1')
                logits, _, _, cor = model(gene_a, gene_b)
                labels_valid = torch.cat((labels_valid, labels))
                logits_valid = torch.cat((logits_valid, logits))
                valid_cor += cor.item()
            valid_loss = criterion(logits_valid, labels_valid).item()
            valid_cor = valid_cor / len(valid_loader)
            valid_auc = roc_auc_score(labels_valid.cpu().detach().numpy(),
                                      logits_valid.cpu().detach().numpy())
            prec, reca, _ = precision_recall_curve(
                labels_valid.cpu().detach().numpy(),
                logits_valid.cpu().detach().numpy())

            valid_f1 = f1_score(
                labels_valid.cpu().detach().numpy(),
                (torch.round(logits_valid)).cpu().detach().numpy())
            valid_aupr = auc(reca, prec)

        with torch.no_grad():
            test_loss = 0
            total_roc = 0
            test_cor = 0
            labels_test = torch.tensor([]).to('cuda:1')
            logits_test = torch.tensor([]).to('cuda:1')
            for i, (gene_a, gene_b, labels) in enumerate(test_loader):
                gene_a, gene_b, labels = gene_a.to('cuda:1'), gene_b.to(
                    'cuda:1'), labels.to('cuda:1')
                logits, _, _, cor = model(gene_a, gene_b)
                labels_test = torch.cat((labels_test, labels))
                logits_test = torch.cat((logits_test, logits))
                test_cor += cor.item()
            test_loss = criterion(logits_test, labels_test).item()
            test_cor = test_cor / len(test_loader)
            test_auc = roc_auc_score(labels_test.cpu().detach().numpy(),
                                     logits_test.cpu().detach().numpy())
            prec, reca, _ = precision_recall_curve(
                labels_test.cpu().detach().numpy(),
                logits_test.cpu().detach().numpy())
            test_f1 = f1_score(
                labels_test.cpu().detach().numpy(),
                (torch.round(logits_test)).cpu().detach().numpy())
            test_aupr = auc(reca, prec)

        print(
            '[Epoch {}] train_loss:{:.4f}, valid_loss:{:.4f}, valid_auc:{:.4f}, valid_aupr:{:.4f}, valid_f1:{:.4f}, test_loss:{:.4f}, test_auc:{:.4f}, test_aupr:{:.4f}, test_f1:{:.4f}, cor:{:.4f}'
            .format(epoch + 1, (running_loss), (valid_loss), (valid_auc),
                    (valid_aupr), (valid_f1), (test_loss), (test_auc),
                    (test_aupr), (test_f1), (train_cor)))
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_valid_aupr = valid_aupr
            best_valid_f1 = valid_f1
            best_test_auc = test_auc
            best_test_aupr = test_aupr
            best_test_f1 = test_f1
            with_loss = test_loss
            with_cor = test_cor
            cnt_wait = 0
        else:
            cnt_wait += 1
        """if (cnt_wait > 20):
            print('Early stopped.')
            break"""
    print('Overall best test_auc:{:.4f}, test_aupr:{:.4f}, test_f1:{:.4f}'.
          format(best_test_auc, best_test_aupr, best_test_f1))

 
