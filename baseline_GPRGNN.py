
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
from torch_geometric.nn import Sequential, GCNConv,GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn import MessagePassing 
import sys

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha]= 1.0
        elif self.Init == 'PPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K
        elif self.Init == 'NPPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3/(self.K+1))
            torch.nn.init.uniform_(self.temp,-bound,bound)
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout , alpha , Init, Gamma , dprate):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

        self.prop1 = GPR_prop(num_layers, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

class GCN_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_2, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList(
            [GCN2Conv(channels = hidden_dim, alpha = 0.1)] +
            [GCN2Conv(channels = hidden_dim, alpha =0.1 ) for _ in range(1, num_layers-1)] + 
            [GCN2Conv(channels = hidden_dim, alpha =0.1 )]
        )
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.zero_layer_exit  = nn.Linear(input_dim, output_dim)

        self.one_layer_exit = GCNConv(input_dim, output_dim)
        # As the dimension of X_0, and H^{(l)} is not the same, we need to change the dimension of X_0 using a linear layer (as explained in the paper Section 3 Paragraph 3)
        self.lin_gcn2_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_gcn2_output = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, edge_index ):
       
        h = x
        if self.num_layers == 0 :
            h = self.zero_layer_exit(h)
            return F.log_softmax(h, dim=1)
        if self.num_layers == 1 :
            h = self.one_layer_exit(h, edge_index)
            return F.log_softmax(h, dim=1)

        x_0_gcn2 = self.lin_gcn2_hidden(x)
        h = x_0_gcn2
        for i, layer in enumerate(self.conv_layers):
  
            if i < len(self.conv_layers) - 1:
                h = layer(h, x_0_gcn2, edge_index)
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)                 

            else : 
                h = layer(h, x_0_gcn2, edge_index)
        h = self.lin_gcn2_output(h)
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
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
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


########################################################################################
# Model Training
########################################################################################
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

    for ex_layer in range(args.num_layers + 1 ) : 
        model = GPRGNN(input_dim=features.shape[1],
                            hidden_dim=args.hidden,
                            output_dim=labels.max().item() + 1,
                            num_layers=ex_layer,
                            dropout=args.dropout, alpha = 0.1  , Init = 'PPR' , Gamma =None  , dprate = 0.5).to(device)

        output = model(features.float(), edge_index )
        print('Training for layer  : ' , ex_layer)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1,args.epochs+1):
            t = time.time()
            model.train()

            optimizer.zero_grad()
            output  = model(features.float(), edge_index )

            loss_train = criterion(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            
            model.eval()
            output  = model(features.float(), edge_index )
            loss_train = criterion(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])

            loss_val = criterion(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            
            print('Epoch: {:04d}'.format(epoch),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t)
                #   'cond: {:.1f}'.format(condition_number(model.gen_adj))
                )


        model.eval()
        output = model(features.float(), edge_index )
        loss_train = criterion(output[idx_train], labels[idx_train])
        train_losses[ex_layer] = train_losses[ex_layer] + [loss_train.item()]

        loss_test = criterion(output[idx_test], labels[idx_test])
        test_losses[ex_layer] = test_losses[ex_layer] + [loss_test.item()]

        train_acc = accuracy(output[idx_train], labels[idx_train])
        train_accuracies[ex_layer] = train_accuracies[ex_layer] + [train_acc.item()]

        test_acc = accuracy(output[idx_test], labels[idx_test])
        test_accuracies[ex_layer] = test_accuracies[ex_layer] + [test_acc.item()]
        print("test_accuracies   :    ", test_accuracies[ex_layer])


for l_ in range(args.num_layers + 1) : 
    print({'mean_train_ll' : np.mean(train_losses[l_]) ,'std_train_ll' : np.std(train_losses[l_]) ,
                'mean_test_ll' : np.mean(test_losses[l_]) ,'std_test_ll' : np.std(test_losses[l_]) ,
                'mean_train_acc' : np.mean(train_accuracies[l_]) ,'std_train_acc' : np.std(train_accuracies[l_]),
                'mean_test_acc' : np.mean(test_accuracies[l_]) ,'std_test_acc' : np.std(test_accuracies[l_]) })
                
