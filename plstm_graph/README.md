# Graph Experiments for pLSTM

Reproduce results with:

```bash
PYTHONPATH=. python train.py --multirun model=tu_bio model_name=plstm batch_size=1 +trainer.accumulate_grad_batches=64 num_layers=2 hidden_dim=96 dataset_name=MUTAG,NCI1,PROTEINS,PTC_FM accelerator=gpu datamodule=tu_datamodule model.graph_pooling_type=mean fold_idx=0,1,2,3,4,5,6,7,8,9 devices=[0]
PYTHONPATH=. python train.py --multirun model=tu_bio model_name=gat,gcn,gin,lstm_gnn,mpnn batch_size=64 num_layers=4,8 hidden_dim=128 dataset_name=MUTAG,NCI1,PROTEINS,PTC_FM accelerator=gpu datamodule=tu_datamodule model.graph_pooling_type=sum fold_idx=0,1,2,3,4,5,6,7,8,9 devices=[0] 
```

Use the root level `requirements.txt` or `environment_jax_torch.yaml` file for the environment installation.