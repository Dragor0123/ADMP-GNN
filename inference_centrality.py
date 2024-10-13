import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GINConv
from torch.nn.init import xavier_uniform_
import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import numpy as np
from utils import condition_number
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import dense_to_sparse
import networkx as nx
from torch_geometric.nn import Sequential, GCNConv
import sys
import pandas as pd



class Normal_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(Normal_GCN, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList(
            [GCNConv(input_dim, hidden_dim)] +
            [GCNConv(hidden_dim, hidden_dim) for _ in range(1, num_layers-1)] + 
            [GCNConv(hidden_dim, output_dim)]
        )
        self.dropout = dropout

        self.zero_layer_exit  = nn.Linear(input_dim, output_dim)

        self.one_layer_exit = GCNConv(input_dim, output_dim)
    def forward(self, x, edge_index ):
        h = x
        if self.num_layers == 0 :
            h = self.zero_layer_exit(h)
            return F.log_softmax(h, dim=1)
        if self.num_layers == 1 :
            h = self.one_layer_exit(h, edge_index)
            return F.log_softmax(h, dim=1)
        for i, layer in enumerate(self.conv_layers):
            h = layer(h, edge_index)
  
            if i < len(self.conv_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)
        # h = self.mlp[i+1](h)
        return F.log_softmax(h, dim=1)
        
# Extern function
import time
import argparse
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
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
import wandb
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
parser.add_argument('--num_clusters', type= int,default = 2)
parser.add_argument('--num_layers', type=int, default=5,  help='Number of hidden units.')
parser.add_argument('--centrality', type=str, default="DEPTH",choices=["DEGREE" , "KCORE", "PAGERANK",'DEPTH']  ,help='which centrality')
parser.add_argument('--dataset', type=str, default='genius', help='Dataset name; default is Cora')
parser.add_argument('--device', type=int, default=0,help='Set CUDA device number; if set to -1, disables cuda.') 
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lr_patience', type=float, default=50, help='Number of epochs waiting for the next lr decay.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')


args = parser.parse_args()
device = torch.device('cuda:'+str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

########################################################################################
# Data loading and model setup 
########################################################################################
if args.dataset == 'genius' or args.dataset == 'Penn94' or args.dataset == 'arxiv-year':
    if args.dataset == 'Penn94' :
        dataset = load_nc_dataset( 'fb100', sub_dataname = args.dataset)
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
    
    adj = to_scipy_sparse_matrix(edge_index)

    G, labels = dataset[0]
    labels = labels.to(device)
    if args.dataset == 'arxiv-year':
        labels = labels.squeeze(1)
    edge_index = dataset.graph['edge_index']    
    adj = to_scipy_sparse_matrix(edge_index)
    G = nx.from_scipy_sparse_matrix(adj)
    print(G)
    
else :
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path = args.datadir, dataset_name = args.dataset,device =  device)
    n = features.size(0)
    G = nx.from_scipy_sparse_matrix(adj)
    print(G)

edge_index, edge_weight = from_scipy_sparse_matrix(adj)
edge_index = edge_index.to(features.device)


########################################################################################
# Model Loading
########################################################################################
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

model = GCN_node_classification(input_dim=features.shape[1],
                            hidden_dim=args.hidden,
                            output_dim=labels.max().item() + 1,
                            num_layers=args.num_layers,
                            dropout=args.dropout).to(device)
test_acc_list = []
val_acc_list = []
for training_id in range(10) :
    checkpoint = torch.load("./checkpoints/checkpoint_{}_training_{}.pt".format(args.dataset ,  training_id))
    model.load_state_dict(checkpoint['model_state_dict'])

    idx_train = checkpoint["idx_train"]
    idx_test = checkpoint["idx_test"]
    idx_val = checkpoint["idx_val"]
    if args.dataset == "genius" :
        X_train= (torch.zeros(n)!=0)
        X_test= (torch.zeros(n) !=0)
        X_val= (torch.zeros(n) !=0)
        X_train[idx_train] = True
        X_test[idx_test] = True
        X_val[idx_val] = True
        idx_train = X_train
        idx_test = X_test
        idx_val = X_val
    
    ######################
    # Degree Centrality
    ######################

    if args.centrality == "DEGREE" :
        degree = {i_:G.degree[i_] for i_ in range(n)}
        centrality = degree
        degree_tensor = torch.tensor([G.degree[i_] for i_ in range(n)]).to(features.device)
        centrality_tensor = degree_tensor
    elif args.centrality == "KCORE" :
        try: 
            core_number_tensor = torch.load( "./centralities/kcore/{}.pth".format(args.dataset)).to(features.device)
        except:
            try:
                core_number_tensor = torch.load( "./centralities/kcore/{}.pth".format(args.dataset)).to(features.device)
            except: 
                core_number = nx.core_number(G)
                centrality = core_number
                core_number_tensor = torch.tensor([core_number[i_] for i_ in range(n)]).to(features.device)
                torch.save(core_number_tensor, "./centralities/kcore/{}.pth".format(args.dataset))  
        centrality_tensor = core_number_tensor
    elif args.centrality == 'PAGERANK' :
        try:
            pr_scores = torch.load( "./centralities/pagerank/{}.pth".format(args.dataset)).to(features.device)
        except:
            try :
                pr_scores = torch.load( "./centralities/pagerank/{}.pth".format(args.dataset)).to(features.device)
            except:
                G = nx.from_scipy_sparse_matrix(adj)
                pr_scores = nx.pagerank(G)
                pr_scores = torch.tensor([pr_scores[k] for k in range(n)]).to(features.device)
                torch.save(pr_scores, "./centralities/pagerank/{}.pth".format(args.dataset))  
        centrality_tensor = pr_scores
    elif args.centrality == 'DEPTH' :
        try:
            n_paths = torch.load( "./centralities/depth/{}.pth".format(args.dataset)).to(device).int()
        except:
            try :
                n_paths = torch.load( "./centralities/depth/{}.pth".format(args.dataset)).to(device)
            except:
                for k_ in range(2) :
                    if k_ == 0 :
                        n_paths = adj
                    else :
                        n_paths = dot(n_paths, adj)
                n_paths = torch.tensor( dot(n_paths, sp.csr_matrix(np.ones((n,1)))).todense()).squeeze(1).to(device)   
                torch.save(n_paths, "./centralities/depth/{}.pth".format(args.dataset))        
        centrality_tensor = n_paths
    centrality_list = centrality_tensor.cpu().numpy().tolist()
    print(centrality_list)
    clusters_set = list(np.arange(args.num_clusters ))
    if args.dataset =="genius" :
        clusters = pd.qcut(centrality_list, args.num_clusters , duplicates='drop')
    else: 
        clusters = pd.qcut(centrality_list, args.num_clusters , labels=[k for k in range(args.num_clusters)],duplicates='drop')
    clusters_set = set(clusters)
    # Compute the best exit for each centrality
    model.eval()
    output = model(features.float(), edge_index ).detach()

    cluster_set_training = set()
    exit_layer_centrality = {}
    cardinal_exit_layer = {}
    for l_ in range(args.num_layers+1) :
        cardinal_exit_layer[l_] = 0
    correct_test_pred = 0
    len_test = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    for c in clusters_set :
        # Training nodes
        c_idx_val= [k_ for k_ in range(len(centrality_list)) if (clusters[k_] == c and idx_val[k_].item()==True)]

        if len(c_idx_val) > 0 :
            cluster_set_training.add(c)
            #loss_per_layer = [criterion(output[:,l_,:][c_idx_val], labels[c_idx_val]).item() for l_ in range(output.size(1))]
            acc_per_layer = [accuracy(output[:,l_,:][c_idx_val], labels[c_idx_val]).item() for l_ in range(output.size(1))]
            print(c , len(c_idx_val))
            print(acc_per_layer)
            #c_exit_layer = np.argwhere(loss_per_layer == np.amin(loss_per_layer)).flatten().tolist()[-1]
            c_exit_layer = np.argwhere(acc_per_layer == np.amax(acc_per_layer)).flatten().tolist()[-1]
            exit_layer_centrality[c] = c_exit_layer
            cardinal_exit_layer[c_exit_layer] = cardinal_exit_layer[c_exit_layer] + len(c_idx_val) 

            # Test nodes
            c_idx_test = [k_ for k_ in range(len(centrality_list)) if (centrality_list[k_] == c and idx_test[k_].item()==True)]
            if len(c_idx_test) > 0 :
                correct_test_pred = correct_test_pred + ( len(c_idx_test) * accuracy(output[:,c_exit_layer,:][c_idx_test], labels[c_idx_test]).item())
                len_test = len_test+len(c_idx_test)

    # Get the most used exit layer for training nodes
    most_used_exit_layers = max(cardinal_exit_layer, key=cardinal_exit_layer.get)
    all_exit = [ ]
    for k in range(n) :
        if clusters[k] in cluster_set_training :
            all_exit.append(exit_layer_centrality[clusters[k]])
        else : 
            all_exit.append(most_used_exit_layers)
    model.eval()
    output  = model(features.float(), edge_index ).detach()
    
    new_val_acc = accuracy(output[[k for k in range(n)],all_exit,:][idx_val], labels[idx_val]).item()
    new_test_acc = accuracy(output[[k for k in range(n)],all_exit,:][idx_test], labels[idx_test]).item()

    val_acc_list.append(new_val_acc)
    test_acc_list.append(new_test_acc)

print("Val Accuracy  : ", np.mean(val_acc_list) , " +- ", np.std(val_acc_list))
print("Test Accuracy  : ", np.mean(test_acc_list) , " +- ", np.std(test_acc_list))

