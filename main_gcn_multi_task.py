# Extern function
import os 
import time
import argparse
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import scipy.sparse as sp
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.utils import degree
import torch.optim as optim
from utils import load_data, load_data_old, accuracy, condition_number
import matplotlib.pyplot as plt
from datetime import datetime
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
from pathlib import Path
import random
import networkx as nx
from numpy import dot
import sys
# Intern function
from models.gcn_conv import GCN_node_classification
from Benchmark.dataset import load_nc_dataset


########################################################################################
# Parse arguments 
########################################################################################

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='./data/',  help='Directory of datasets; default is ./data/')
parser.add_argument('--num_layers', type=int, default=5,  help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name; default is Cora')
parser.add_argument('--device', type=int, default=0,help='Set CUDA device number; if set to -1, disables cuda.') 
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lr_patience', type=float, default=50, help='Number of epochs waiting for the next lr decay.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')


args = parser.parse_args()
device = torch.device('cuda:'+str(args.device)) if torch.cuda.is_available() else torch.device('cpu')


if args.dataset == 'ogbn-arxiv' :
    args.hidden = 512
if args.dataset == 'Cora' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.8

elif args.dataset == 'CiteSeer' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.4
elif args.dataset == 'PubMed' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.2
elif args.dataset == 'CS' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.4
elif args.dataset == 'genius' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.8
elif args.dataset == 'Penn94' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.2
elif args.dataset == 'Computers' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2
elif args.dataset == 'Photo' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.6
elif args.dataset == 'Physics' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.4
elif args.dataset == 'twitch-gamers' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2   
elif args.dataset == 'deezer-europe' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2     
elif args.dataset == 'imdb' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2     
elif args.dataset == 'chameleon' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2  
elif args.dataset == "Cornell" or args.dataset ==  "Texas" or args.dataset ==  "Wisconsin" :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2  

    
train_losses = {}
test_losses = {}
train_accuracies = {}
test_accuracies = {}

for l_ in range(args.num_layers + 1 ) :
    train_losses[l_] = []
    test_losses[l_] = []
    train_accuracies[l_] = []
    test_accuracies[l_] = []

for training in range(10) :

    ########################################################################################
    # Data loading and model setup 
    ########################################################################################
    if args.dataset == 'genius' or args.dataset == 'Penn94' or args.dataset == 'arxiv-year':
        if args.dataset == 'Penn94' :
            dataset = load_nc_dataset( 'fb100'  , sub_dataname = args.dataset)
        else : 
            dataset = load_nc_dataset( args.dataset  , sub_dataname='')
        features = dataset.graph['node_feat'].to(device)
        n = features.size(0)
        split_idx = dataset.get_idx_split(split_type='random', train_prop=.5, valid_prop=.25)  # By default  train_prop=.5, valid_prop=.25
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        G, labels = dataset[0]
        labels = labels.to(device)
        if args.dataset == 'arxiv-year':
            labels = labels.squeeze(1)
        edge_index = dataset.graph['edge_index']
        print(G)
        adj = to_scipy_sparse_matrix(edge_index)
        G = nx.from_scipy_sparse_matrix(adj)
    else :
        adj, features, labels, idx_train, idx_val, idx_test = load_data(path = args.datadir, dataset_name = args.dataset,device =  device)
        n = features.size(0)
        G = nx.from_scipy_sparse_matrix(adj)
        print(G)

    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    edge_index = edge_index.to(features.device)
    ########################################################################################
    # Model Training
    ########################################################################################

    model = GCN_node_classification(input_dim=features.shape[1],
                            hidden_dim=args.hidden,
                            output_dim=labels.max().item() + 1,
                            num_layers=args.num_layers,
                            dropout=args.dropout).to(device)

    output = model(features.float(), edge_index )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1,args.epochs+1):
        t = time.time()
        model.train()

        optimizer.zero_grad()
        output  = model(features.float(), edge_index )
        loss_train = 0.0
        for l_ in range(output.size(1)) : 
            loss_train = loss_train + criterion(output[:,l_,:][idx_train], labels[idx_train])
        loss_train = loss_train/output.size(1)
        loss_train.backward()
        optimizer.step()
        
    model.eval() 
    output  = model(features.float(), edge_index )
    for ex_layer in range(output.size(1)) :
        loss_train = criterion(output[:,ex_layer,:][idx_train], labels[idx_train])
        train_losses[ex_layer] = train_losses[ex_layer] + [loss_train.item()]

        loss_test = criterion(output[:,ex_layer,:][idx_test], labels[idx_test])
        test_losses[ex_layer] = test_losses[ex_layer] + [loss_test.item()]

        train_acc = accuracy(output[:,ex_layer,:][idx_train], labels[idx_train])
        train_accuracies[ex_layer] = train_accuracies[ex_layer] + [train_acc.item()]

        test_acc = accuracy(output[:,ex_layer,:][idx_test], labels[idx_test])
        test_accuracies[ex_layer] = test_accuracies[ex_layer] + [test_acc.item()]
