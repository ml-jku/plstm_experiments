import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, BinaryAUROC, MultilabelAUROC, MulticlassAUROC, AveragePrecision, F1Score
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.aggregation import MeanMetric
#from ogb.lsc import PCQM4Mv2Evaluator
import networkx as nx
import random


class MultiClass:
    def __init__(self, num_classes, metrics, main_metric, metric_mode, external_eval=None):
        self.metrics = {}
        for i in ['train', 'valid', 'test']:
            self.metrics[i] = {}
            self.metrics[i]['loss'] = MeanMetric()
            if 'acc' in metrics:
                self.metrics[i]['acc'] = Accuracy(task='multiclass', num_classes=num_classes)
            if 'roc_auc' in metrics:
                self.metrics[i]['roc_auc'] = MulticlassAUROC(num_classes, average='macro')
            if 'avg_precision' in metrics:
                self.metrics[i]['avg_precision'] = AveragePrecision(task='multilabel', num_labels=num_classes)
            if 'f1' in metrics:
                self.metrics[i]['f1'] = F1Score(task='multiclass', num_classes=num_classes, average='macro')

            self.external_eval = external_eval

            self.main_metric = main_metric
            self.metric_mode = metric_mode

            if external_eval is not None:
                self.cat_metrics = {
                    "train": {"y_true": [],
                              "y_pred": []},
                    "valid": {"y_true": [],
                              "y_pred": []},
                    "test": {"y_true": [],
                             "y_pred": []}
                }

    def step(self, x, batch, loss_func, mode):
        y = batch.y
        loss = loss_func(x, y)
        _, x_acc = torch.max(x, 1)

        if 'acc' in self.metrics[mode]:
            self.metrics[mode]['acc'].update(x_acc, y)
        x_auc = F.softmax(x, dim=-1)
        if 'roc_auc' in self.metrics[mode]:
            self.metrics[mode]['roc_auc'].update(x_auc, y)
        if 'avg_precision' in self.metrics[mode]:
            self.metrics[mode]['roc_auc'].update(x_auc, y)
        if 'f1' in self.metrics[mode]:
            self.metrics[mode]['f1'].update(x_auc, y)

        self.metrics[mode]['loss'].update(loss)
        if self.external_eval is not None:
            #if len(y.shape) < len(x.shape):
            #    _y = y.detach().to('cpu').unsqueeze(dim=1)
            #else:
            #    _y = y.detach().to('cpu')
            self.cat_metrics[mode]['y_true'].append(y.detach().to('cpu'))
            self.cat_metrics[mode]['y_pred'].append(x.detach().to('cpu'))

        return loss

class MultiClassMask:
    def __init__(self, num_classes, metrics, main_metric, metric_mode, external_eval=None):
        self.metrics = {}
        for i in ['train', 'valid', 'test']:
            self.metrics[i] = {}
            self.metrics[i]['loss'] = MeanMetric()
            if 'acc' in metrics:
                self.metrics[i]['acc'] = Accuracy(task='multiclass', num_classes=num_classes)
            if 'roc_auc' in metrics:
                self.metrics[i]['roc_auc'] = MulticlassAUROC(num_classes, average='macro')
            if 'avg_precision' in metrics:
                self.metrics[i]['avg_precision'] = AveragePrecision(task='multilabel', num_labels=num_classes)
            if 'f1' in metrics:
                self.metrics[i]['f1'] = F1Score(task='multiclass', num_classes=num_classes, average='macro')

            self.external_eval = external_eval

            self.main_metric = main_metric
            self.metric_mode = metric_mode

            if external_eval is not None:
                self.cat_metrics = {
                    "train": {"y_true": [],
                              "y_pred": []},
                    "valid": {"y_true": [],
                              "y_pred": []},
                    "test": {"y_true": [],
                             "y_pred": []}
                }

    def step(self, x, batch, loss_func, mode):
        y = batch.y
        # apply mask
        x = x[batch[mode + "_mask"]]
        y = y[batch[mode + "_mask"]]

        loss = loss_func(x, y)
        _, x_acc = torch.max(x, 1)

        if 'acc' in self.metrics[mode]:
            self.metrics[mode]['acc'].update(x_acc, y)
        x_auc = F.softmax(x, dim=-1)
        if 'roc_auc' in self.metrics[mode]:
            self.metrics[mode]['roc_auc'].update(x_auc, y)
        if 'avg_precision' in self.metrics[mode]:
            self.metrics[mode]['roc_auc'].update(x_auc, y)
        if 'f1' in self.metrics[mode]:
            self.metrics[mode]['f1'].update(x_auc, y)

        self.metrics[mode]['loss'].update(loss)
        if self.external_eval is not None:
            #if len(y.shape) < len(x.shape):
            #    _y = y.detach().to('cpu').unsqueeze(dim=1)
            #else:
            #    _y = y.detach().to('cpu')
            self.cat_metrics[mode]['y_true'].append(y.detach().to('cpu'))
            self.cat_metrics[mode]['y_pred'].append(x.detach().to('cpu'))

        return loss


class MultiLabel:
    def __init__(self, num_classes, metrics, main_metric, metric_mode, external_eval=None):
        self.metrics = {}
        for i in ['train', 'valid', 'test']:
            self.metrics[i] = {}
            self.metrics[i]['loss'] = MeanMetric()
            if 'acc' in metrics:
                if num_classes == 1:
                    self.metrics[i]['acc'] = Accuracy(task='binary')
                else:
                    self.metrics[i]['acc'] = Accuracy(task='multilabel', num_labels=num_classes)
            if 'roc_auc' in metrics:
                if num_classes == 1:
                    self.metrics[i]['roc_auc'] = BinaryAUROC()
                else:
                    self.metrics[i]['roc_auc'] = MultilabelAUROC(num_classes, average='macro')
            if 'avg_precision' in metrics:
                self.metrics[i]['avg_precision'] = AveragePrecision(task='multilabel', num_labels=num_classes)

            self.external_eval = external_eval

            self.main_metric = main_metric
            self.metric_mode = metric_mode

            if external_eval is not None:
                self.cat_metrics = {
                    "train": {"y_true": [],
                              "y_pred": []},
                    "valid": {"y_true": [],
                              "y_pred": []},
                    "test": {"y_true": [],
                             "y_pred": []}
                }

    def step(self, x, batch, loss_func, mode):
        y = batch.y
        loss = loss_func(x, y)

        y = y.to(torch.long)

        x = F.sigmoid(x)

        if 'acc' in self.metrics[mode]:
            self.metrics[mode]['acc'].update(x, y)
        if 'roc_auc' in self.metrics[mode]:
            self.metrics[mode]['roc_auc'].update(x, y)
        if 'avg_precision' in self.metrics[mode]:
            self.metrics[mode]['avg_precision'].update(x, y)

        self.metrics[mode]['loss'].update(loss)
        if self.external_eval is not None:
            self.cat_metrics[mode]['y_true'].append(y.detach().to('cpu'))
            self.cat_metrics[mode]['y_pred'].append(x.detach().to('cpu'))

        return loss


class Regression:
    def __init__(self, num_classes, metrics, main_metric, metric_mode, external_eval=None):
        self.metrics = {}
        for i in ['train', 'valid', 'test']:
            self.metrics[i] = {}
            self.metrics[i]['loss'] = MeanMetric()
            if 'mae' in metrics:
                self.metrics[i]['mae'] = MeanAbsoluteError()
            if 'mse' in metrics:
                self.metrics[i]['mse'] = MeanSquaredError()

            self.external_eval = external_eval

            self.main_metric = main_metric
            self.metric_mode = metric_mode

            if external_eval is not None:
                self.cat_metrics = {
                    "train": {"y_true": [],
                              "y_pred": []},
                    "valid": {"y_true": [],
                              "y_pred": []},
                    "test": {"y_true": [],
                             "y_pred": []}
                }

    def step(self, x, batch, loss_func, mode):
        y = batch.y
        loss = loss_func(x, y)

        for i in self.metrics[mode]:
            if i != 'loss':
                self.metrics[mode][i].update(x, y)

        self.metrics[mode]['loss'].update(loss)

        if self.external_eval is not None:
            #if isinstance(self.external_eval, PCQM4Mv2Evaluator):
            #    _x = x.detach().to('cpu').squeeze(dim=1)
            #    _y = y.detach().to('cpu').squeeze(dim=1)
            #else:
            _x = x.detach().to('cpu')
            _y = y.detach().to('cpu')
            self.cat_metrics[mode]['y_true'].append(_y)
            self.cat_metrics[mode]['y_pred'].append(_x)

        return loss