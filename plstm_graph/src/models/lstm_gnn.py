import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from src.base_model import BaseModel


class NodeConv(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.root_conv_z = nn.Linear(node_dim, node_dim, bias=True)
        self.rel_conv_z = nn.Linear(node_dim, node_dim, bias=False)
        self.root_conv_i = nn.Linear(node_dim, node_dim, bias=True)
        self.rel_conv_i = nn.Linear(node_dim, node_dim, bias=False)
        self.root_conv_f = nn.Linear(node_dim, node_dim, bias=True)
        self.rel_conv_f = nn.Linear(node_dim, node_dim, bias=False)
        self.root_conv_o = nn.Linear(node_dim, node_dim, bias=True)
        self.rel_conv_o = nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, h, c, row, col, batch, neighbor_norm=None):
        agg = scatter(h[row], col, dim=0, dim_size=h.size(0), reduce='sum')
        if neighbor_norm is not None:
            agg = agg / neighbor_norm

        rel_z = self.rel_conv_z(agg)
        root_z = self.root_conv_z(h)
        z = F.tanh(rel_z + root_z)

        rel_i = self.rel_conv_i(agg)
        root_i = self.root_conv_i(h)
        i = F.sigmoid(rel_i + root_i)

        rel_f = self.rel_conv_f(agg)
        root_f = self.root_conv_f(h)
        f = F.sigmoid(rel_f + root_f)

        rel_o = self.rel_conv_o(agg)
        root_o = self.root_conv_o(h)
        o = F.sigmoid(rel_o + root_o)

        c = f * c + i * z
        h = o * F.tanh(c)
        return h, c


class GNNModel(BaseModel):
    def __init__(self, task, num_layers, num_features, hidden_dim, num_classes, rand_features, graph_pooling_type,
                 neighbor_pooling_type, node_encoder, edge_encoder, lr, loss, evaluator, optimizer=None):
        super(GNNModel, self).__init__(task, num_layers, num_features, hidden_dim, num_classes, rand_features,
                                       graph_pooling_type, neighbor_pooling_type, node_encoder, edge_encoder,
                                       lr, loss, evaluator, optimizer)

        self.node_conv = NodeConv(hidden_dim)

    def forward(self, data, mode=None):
        x, edge_index, edge_attr, batch = (data.x.to(self.device),
                                           data.edge_index.to(self.device),
                                           data.edge_attr,
                                           data.batch.to(self.device))
        row, col = edge_index

        x = self.node_encoder(x)

        if self.rand_features:
            rand = torch.randn_like(x)
            x = x + rand

        c = x
        # no input from previous layers
        h = torch.zeros_like(c)

        if self.rand_features:
            rand = torch.randn_like(c)
            c = c + rand

        # is the same for all layers
        adj_norm = self.adj_normalization(x, row, col)

        for layer in range(self.num_layers):
            h, c = self.node_conv(h, c, row, col, batch, adj_norm)

        c = F.tanh(c)

        if self.pool is not None:
            c = self.pool(c, batch)

        x = self.decoder(c)

        return x