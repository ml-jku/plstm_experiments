import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation
from torch_geometric.utils import scatter

from src.utils.scheduler import get_cosine_schedule_with_warmup


class BaseModel(pl.LightningModule):
    def __init__(self, task, num_layers, num_features, hidden_dim, num_classes, rand_features,
                 graph_pooling_type, neighbor_pooling_type, node_encoder, edge_encoder, lr, loss, evaluator, optimizer):
        super(BaseModel, self).__init__()

        self.task = task
        self.num_layers = num_layers
        self.lr = lr

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.rand_features = rand_features
        self.num_classes = num_classes

        self.neighbor_pooling_type = neighbor_pooling_type
        self.graph_pooling_type = graph_pooling_type

        if self.graph_pooling_type == 'sum':
            self.pool = SumAggregation()
        elif self.graph_pooling_type == 'mean':
            self.pool = MeanAggregation()
        else:
            self.pool = None

        self.node_conv = None

        self.neighbor_encoding = False
        if node_encoder is None:
            self.node_encoder = nn.Linear(num_features, hidden_dim)
        else:
            self.node_encoder = node_encoder

        self.edge_encoder = edge_encoder

        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, num_classes))
        self.loss = loss
        self.evaluator = evaluator
        self.opt = optimizer

    def adj_normalization(self, x, row, col):
        # one for every row/col
        shape = [len(row), 1]
        ones_like_edges = torch.ones(size=shape, dtype=x.dtype, device=x.device)

        if self.neighbor_pooling_type in ('vpa', 'mean'):
            # count the number of neighbors
            adj_norm = scatter(ones_like_edges, col, dim=0, dim_size=x.size(0), reduce='sum')
            adj_norm = torch.clamp(adj_norm, min=1)
            if self.neighbor_pooling_type == 'vpa':
                # apply sqrt for variance presverving aggregation
                adj_norm = torch.sqrt(adj_norm)
        else:
            adj_norm = None

        return adj_norm

    def process_step(self, batch, batch_idx, mode):
        x = self(batch, mode)
        loss = self.evaluator.step(x, batch, self.loss, mode)

        return loss

    def on_train_epoch_start(self):
        for i in self.evaluator.metrics["train"]:
            self.evaluator.metrics["train"][i].to(self.device)

    def on_validation_epoch_start(self):
        for i in self.evaluator.metrics["valid"]:
            self.evaluator.metrics["valid"][i].to(self.device)

    def on_test_epoch_start(self):
        for i in self.evaluator.metrics["test"]:
            self.evaluator.metrics["test"][i].to(self.device)

    def training_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "test")

    def on_epoch_end(self, mode):
        for i in self.evaluator.metrics[mode]:
            self.log(f"{mode}/" + i, self.evaluator.metrics[mode][i].compute())
            self.evaluator.metrics[mode][i].reset()
        if self.evaluator.external_eval is not None:
            y_true = torch.cat(self.evaluator.cat_metrics[mode]["y_true"], dim=0)
            y_pred = torch.cat(self.evaluator.cat_metrics[mode]["y_pred"], dim=0)
            eval = self.evaluator.external_eval.eval({"y_true": y_true, "y_pred": y_pred})

            if len(eval) == 1:
                for i in eval:
                    self.log(f"{mode}/eval", eval[i])
            else:
                raise NotImplementedError("multiple evals exist")
            for i in self.evaluator.cat_metrics[mode]:
                self.evaluator.cat_metrics[mode][i] = []

    def on_train_epoch_end(self):
        self.on_epoch_end("train")
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0])

    def on_validation_epoch_end(self):
        self.on_epoch_end("valid")

    def on_test_epoch_end(self):
        self.on_epoch_end("test")

    def configure_optimizers(self):
        if self.opt is None:
            optim = torch.optim.AdamW(self.parameters(), self.lr)
            scheduler = get_cosine_schedule_with_warmup(optim, int(self.trainer.max_epochs / 20), self.trainer.max_epochs)
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler}}
