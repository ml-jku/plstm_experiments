import torch
import torch.nn as nn
from src.base_model import BaseModel
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import scatter


class GNNModel(BaseModel):
    def __init__(self, task, num_layers, num_features, hidden_dim, num_classes, rand_features, graph_pooling_type,
                 neighbor_pooling_type, node_encoder, edge_encoder, lr, loss, evaluator, optimizer=None):
        super(GNNModel, self).__init__(task, num_layers, num_features, hidden_dim, num_classes, rand_features,
                                       graph_pooling_type, neighbor_pooling_type, node_encoder, edge_encoder,
                                       lr, loss, evaluator, optimizer)

        self.node_convs = nn.ModuleList()

        for i in range(num_layers):
            self.node_convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, data, mode=None):
        x, edge_index, edge_attr, batch = (data.x.to(self.device),
                                           data.edge_index.to(self.device),
                                           data.edge_attr,
                                           data.batch.to(self.device))

        x = self.node_encoder(x)

        if self.rand_features:
            rand = torch.randn_like(x)
            x = x + rand

        for layer in range(self.num_layers):
            x = self.node_convs[layer](x, edge_index)
            x = F.relu(x)

        if self.pool is not None:
            x = self.pool(x, batch)

        x = self.decoder(x)

        return x
