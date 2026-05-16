import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool


class GCNNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCNNet, self).__init__()
        
        self.args = args
        self.n_cls = len(dataset.classdict)
        self.conv1 = GCNConv(dataset.data.x.shape[1], args.dim)
        self.conv2 = GCNConv(args.dim, args.dim)
        self.fc = Linear(args.dim, self.n_cls)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        z = global_mean_pool(x, data.batch)
        x = self.fc(z)
        x = F.dropout(x, training=self.training)
        return x, z
    

class MDGCNNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MDGCNNet, self).__init__()
        
        self.args = args
        self.n_cls = len(dataset.classdict)
        self.conv1 = GCNConv(dataset.data.x.shape[1], args.dim)
        self.conv2 = GCNConv(args.dim, args.dim)
        self.fc = Linear(args.dim, self.args.com*(self.n_cls+1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        pi_mu = self.fc(x)
        pi_mu = F.dropout(pi_mu, training=self.training)
        pi = pi_mu[:, :self.args.com]
        mu = pi_mu[:, self.args.com:].reshape(-1, self.args.com, self.n_cls)
        return pi, mu


class GATNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GATNet, self).__init__()
        
        self.args = args
        self.n_cls = len(dataset.classdict)
        self.conv1 = GATConv(dataset.data.x.shape[1], args.dim)
        self.conv2 = GATConv(args.dim, args.dim)
        self.fc = Linear(args.dim, self.n_cls)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        z = global_mean_pool(x, data.batch)
        x = self.fc(z)
        x = F.dropout(x, training=self.training)
        return x, z
    

class MDGATNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MDGATNet, self).__init__()
        
        self.args = args
        self.n_cls = len(dataset.classdict)
        self.conv1 = GATConv(dataset.data.x.shape[1], args.dim)
        self.conv2 = GATConv(args.dim, args.dim)
        self.fc = Linear(args.dim, self.args.com*(self.n_cls+1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        pi_mu = self.fc(x)
        pi_mu = F.dropout(pi_mu, training=self.training)
        pi = pi_mu[:, :self.args.com]
        mu = pi_mu[:, self.args.com:].reshape(-1, self.args.com, self.n_cls)
        return pi, mu
    

class GINNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GINNet, self).__init__()
        
        self.args = args
        self.n_cls = len(dataset.classdict)
        self.conv1 = GINConv(nn.Linear(dataset.data.x.shape[1], args.dim))
        self.conv2 = GINConv(nn.Linear(args.dim, args.dim))
        self.fc = Linear(args.dim, self.n_cls)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        z = global_mean_pool(x, data.batch)
        x = self.fc(z)
        x = F.dropout(x, training=self.training)
        return x, z
    

class MDGINNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MDGINNet, self).__init__()
        
        self.args = args
        self.n_cls = len(dataset.classdict)
        self.conv1 = GINConv(nn.Linear(dataset.data.x.shape[1], args.dim))
        self.conv2 = GINConv(nn.Linear(args.dim, args.dim))
        self.fc = Linear(args.dim, self.args.com*(self.n_cls+1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        pi_mu = self.fc(x)
        pi_mu = F.dropout(pi_mu, training=self.training)
        pi = pi_mu[:, :self.args.com]
        mu = pi_mu[:, self.args.com:].reshape(-1, self.args.com, self.n_cls)
        return pi, mu