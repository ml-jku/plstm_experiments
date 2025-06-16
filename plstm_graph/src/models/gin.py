import torch
import torch.nn as nn
from src.base_model import BaseModel
from torch_geometric.nn import GINConv
import torch.nn.functional as F
from torch_geometric.utils import scatter


def build_mlp(num_mlp_layers, num_features, hidden_dim):
    layers = []
    layers.append(nn.Linear(num_features, hidden_dim))

    for _ in range(num_mlp_layers - 1):
        layers.append(nn.BatchNorm1d((hidden_dim)))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))

    return nn.Sequential(*layers)


class GNNModel(BaseModel):
    def __init__(self, task, num_layers, num_features, hidden_dim, num_classes, rand_features, graph_pooling_type,
                 neighbor_pooling_type, node_encoder, edge_encoder, lr, loss, evaluator, optimizer=None):
        super(GNNModel, self).__init__(task, num_layers, num_features, hidden_dim, num_classes, rand_features,
                                       graph_pooling_type, neighbor_pooling_type, node_encoder, edge_encoder,
                                       lr, loss, evaluator, optimizer)


        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            mlp = build_mlp(2, hidden_dim, hidden_dim)
            conv = GINConv(nn=mlp)

            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.linears_prediction.append(nn.Linear(hidden_dim, num_classes))
        self.linears_prediction.append(nn.Linear(hidden_dim, num_classes))

    def forward(self, data, mode=None):
        x, edge_index, edge_attr, batch = (data.x.to(self.device),
                                           data.edge_index.to(self.device),
                                           data.edge_attr,
                                           data.batch.to(self.device))
        x = self.node_encoder(x)

        if self.rand_features:
            rand = torch.randn_like(x)
            x = x + rand

        hiddens = [x]
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
            hiddens.append(x)

        score_over_layer = 0
        for layer, h in enumerate(hiddens):
            if self.pool is not None:
                h = self.pool(h, batch)
            score_over_layer += self.linears_prediction[layer](h)

        #x = self.decoder(x)

        return score_over_layer

