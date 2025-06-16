import torch
from torch import nn
from src.base_model import BaseModel
from plstm.torch.plstm_graph_layer import PreparedGraph
from plstm.torch.graph_block import pLSTMGraphEdgeBlockConfig, pLSTMGraphEdgeBlock
from torch_geometric.utils import to_networkx, scatter


class GNNModel(BaseModel):
    def __init__(
        self,
        task,
        num_layers,
        num_features,
        hidden_dim,
        num_classes,
        rand_features,
        graph_pooling_type,
        neighbor_pooling_type,
        node_encoder,
        edge_encoder,
        lr,
        loss,
        evaluator,
        optimizer=None,
        max_edges=4,
        num_heads=4,
    ):
        super(GNNModel, self).__init__(
            task,
            num_layers,
            num_features,
            hidden_dim,
            num_classes,
            rand_features,
            graph_pooling_type,
            neighbor_pooling_type,
            node_encoder,
            edge_encoder,
            lr,
            loss,
            evaluator,
            optimizer,
        )

        self.mode = task
        self.max_edges = max_edges
        self.num_heads = num_heads
        self.node_convs = nn.ModuleList()
        for i in range(num_layers):
            cfg = pLSTMGraphEdgeBlockConfig(hidden_dim, num_heads=self.num_heads)
            block = pLSTMGraphEdgeBlock(cfg)
            self.node_convs.append(block)

    def forward(self, data, mode=None):
        x, edge_index, edge_attr, batch = (
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.edge_attr,
            data.batch.to(self.device),
        )
        row, col = edge_index

        nx_graph = to_networkx(data)

        graph_D = PreparedGraph.create(nx_graph, mode="D")
        graph_P = PreparedGraph.create(nx_graph, mode="P")
        graph_P_r = graph_P.reverse(graph_P)
        graph_D_r = graph_D.reverse(graph_P)

        x = self.node_encoder(x)

        x = self.encode_neighbor(row, col, x)

        if self.edge_encoder is None:
            edge_x = (x[row] + x[col]) / 2
        else:
            edge_x = self.edge_encoder(edge_attr)

        if self.rand_features:
            rand = torch.randn_like(x)
            x = x + rand

        for layer in range(self.num_layers):
            node_conv = self.node_convs[layer]
            if layer % 2 == 0:
                x = node_conv(x, edge_features=edge_x, graph_P=graph_P, graph_D=graph_D)
            else:
                x = node_conv(x, edge_features=edge_x, graph_P=graph_P_r, graph_D=graph_D_r)

        if self.pool is not None:
            x = self.pool(x, batch)

        x = self.decoder(x)

        return x

    def encode_neighbor(self, row, col, x):
        # one for every row/col
        shape = [len(row), 1]
        ones_like_edges = torch.ones(size=shape, dtype=x.dtype, device=x.device)
        # count the number of neighbours
        num_neighbors = scatter(ones_like_edges, col, dim=0, dim_size=x.size(0), reduce="sum")
        # positional encoding
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2).float() * (-torch.log(torch.ones(1) * 10000) / self.hidden_dim)
        ).to(self.device)
        positional_encoding = torch.zeros(len(num_neighbors), self.hidden_dim)
        positional_encoding[:, 0::2] = torch.sin(num_neighbors * div_term)
        positional_encoding[:, 1::2] = torch.cos(num_neighbors * div_term)
        x = x + positional_encoding.to(self.device)
        return x
