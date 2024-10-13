################################
# Convolutional models
################################


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GINConv
from torch.nn.init import xavier_uniform_
import torch
import numpy as np
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import dense_to_sparse
import networkx as nx
#from torch_geometric.nn import Sequential, GCNConv, GATConv
from models.custom import AGG_Conv
import sys



class GCN_node_classification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_node_classification, self).__init__()
        self.num_layers = num_layers
        self.aggregates = nn.ModuleList(
            [AGG_Conv()] +
            [AGG_Conv() for _ in range(1, num_layers-1)] + 
            [AGG_Conv()]
        )
        self.updates_continues = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] +
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(1, num_layers-1)] + 
            [nn.Linear(hidden_dim, output_dim)]
        )
        # The number of layers is aqual to the number of GCNConv + 1 (a node can exit at the early biginning)
        self.updates_exit = nn.ModuleList(
            [nn.Linear(input_dim, output_dim)] +
            [nn.Linear(input_dim, output_dim)] +
            [nn.Linear(hidden_dim, output_dim) for _ in range(1, num_layers-1)] + 
            [nn.Linear(hidden_dim, output_dim)]
        )

        self.dropout = dropout
    def forward(self, x, edge_index ):
        h = x
        h_exit = self.updates_exit[0](h)
        output = F.log_softmax(h_exit, dim=1).unsqueeze(1)

        for i, layer in enumerate(self.aggregates):
            h = layer(h, edge_index)
            h_exit = self.updates_exit[i+1](h)
            output = torch.cat([output , F.log_softmax(h_exit, dim=1).unsqueeze(1)] , dim = 1)
            h = self.updates_continues[i](h)
            if i < len(self.aggregates) - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)

        return output

