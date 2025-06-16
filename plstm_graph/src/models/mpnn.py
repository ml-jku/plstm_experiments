import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from src.base_model import BaseModel


class NodeConv(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.root_conv = nn.Linear(node_dim, node_dim, bias=True)
        self.rel_conv = nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, x, row, col, batch, adj_norm=None):
        agg = scatter(x[row], col, dim=0, dim_size=x.size(0), reduce='sum')
        if adj_norm is not None:
            agg = agg / adj_norm
        rel = self.rel_conv(agg)
        root = self.root_conv(x)
        out = rel + root
        out = F.relu(out)
        return out


class GNNModel(BaseModel):
    def __init__(self, task, num_layers, num_features, hidden_dim, num_classes, rand_features, graph_pooling_type,
                 neighbor_pooling_type, node_encoder, edge_encoder, lr, loss, evaluator, optimizer=None):
        super(GNNModel, self).__init__(task, num_layers, num_features, hidden_dim, num_classes, rand_features,
                                       graph_pooling_type, neighbor_pooling_type, node_encoder, edge_encoder,
                                       lr, loss, evaluator, optimizer)

        self.node_convs = nn.ModuleList()

        for i in range(num_layers):
            self.node_convs.append(NodeConv(hidden_dim))

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

        # is the same for all layers
        adj_norm = self.adj_normalization(x, row, col)

        for layer in range(self.num_layers):
            node_conv = self.node_convs[layer]
            x = node_conv(x, row, col, batch, adj_norm)

        if self.pool is not None:
            x = self.pool(x, batch)

        x = self.decoder(x)

        return x