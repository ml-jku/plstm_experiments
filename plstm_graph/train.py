import os
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from hydra.utils import instantiate
from omegaconf import DictConfig
from copy import deepcopy
import torch
import wandb
import sys

torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def run(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    print("PYTHONPATH", sys.path)
    print("Working directory : {}".format(os.getcwd()))
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = instantiate(cfg.logger)(config=config)

    try:
        logger.experiment.save(f"./src/models/{cfg.model_name}.py")
    except OSError:
        pass

    datamodule = instantiate(cfg.datamodule)
    dataset = datamodule.train_dataloader().dataset

    try:
        if dataset.num_features == 0:
            cfg.model.num_features = 1
        else:
            cfg.model.num_features = dataset.num_features
    except:
        pass

    if cfg.model.num_classes == 0:
        cfg.model.num_classes = dataset.num_classes
        cfg.model.evaluator.num_classes = dataset.num_classes

    model = instantiate(cfg.model)

    main_metric = "acc"
    metric_mode = "max"
    checkpoint_callback = ModelCheckpoint(monitor=f"valid/{main_metric}", mode=metric_mode, save_top_k=1)
    trainer: pl.Trainer = instantiate(cfg.trainer, logger=logger, callbacks=[checkpoint_callback])

    print("model parameters", sum([p.numel() for p in model.parameters()]))

    try:
        trainer.validate(deepcopy(model), datamodule.val_dataloader())
    except:
        pass
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    run()
