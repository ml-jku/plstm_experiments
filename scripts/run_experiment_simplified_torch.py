#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Any
import itertools

import subprocess
import tqdm
import random

from omegaconf import OmegaConf

### changed for torch
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

###
import os
import numpy as np

from compoconf import parse_config
import sys
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent))
from src import extract_hydra
from src.extract_hydra import run_hydra
from src.simple_training import (
    VisionTrainerConfig,
    ### changed for torch
    # TrainState,
    # create_train_state,
    # training_step,
    # validation_step,
    # create_train_nnx_state,
    # training_nnx_step,
    # validation_nnx_step,
    ###
)
from src.simple_training_torch import (
    TorchTrainState,
    create_torch_train_state,
    torch_training_step,
    torch_validation_step,
)
from src.simple_dataset import SimpleDatasetModule
from src.utils import LimitIterable, AverageMeter, save_checkpoint_in_background

import src.simple_dataset  # noqa

### changed for torch
# import src.model_wrappers_linen  # noqa
# import src.model_wrappers  # noqa
# from jax_trainer.interfaces import BaseModelLinen, BaseModelNNX
import src.model_wrappers_torch  # noqa
from src.model_wrappers_torch import BaseModelTorch
###

### changed for torch
from src.optimizers_torch import OptimizerInterface

###
import wandb
### changed for torch
# from flax.jax_utils import unreplicate
# from flax.serialization import msgpack_serialize
# from flax.training.common_utils import shard
# from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
# from jax.experimental import mesh_utils
# from flax import nnx
###

import logging


LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
LOG_FORMAT = "[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s]{rank} - %(message)s"
stdout_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    handlers=[stdout_handler],
    level=LOGLEVEL,
    format=LOG_FORMAT.format(rank=""),
    force=True,
)
LOGGER = logging.getLogger(__name__)


def exception_logging(typ, value, traceback):
    LOGGER.exception("Uncaught exception", exc_info=(typ, value, traceback))


sys.excepthook = exception_logging


__spec__ = None


### changed for torch
def init_distributed():
    """Initialize distributed training."""
    if "SLURM_NTASKS" in os.environ and int(os.environ["SLURM_NTASKS"]) > 1:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))

        # Initialize process group
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    elif torch.cuda.device_count() > 1:
        # Multi-GPU single node
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0")

        # For single node multi-GPU, we'll use DataParallel instead of DDP
        # dist.init_process_group(backend="nccl", init_method="env://")

    else:
        # Single GPU or CPU
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, device


###


@dataclass
class ExperimentConfig:
    trainer: VisionTrainerConfig
    ### changed for torch
    model: BaseModelTorch.cfgtype
    ###
    optimizer: OptimizerInterface.cfgtype
    dataset: SimpleDatasetModule.cfgtype
    ### changed for torch
    # batch_mesh_axis: Any
    ###
    slurm: Any = None
    env: Any = None
    config_unresolved: Any = None
    config_args: Any = None
    seed: int = 42
    aux: dict[str, Any] = field(default_factory=dict)


### changed for torch
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, prefix="val/") -> dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    average_meter = AverageMeter()

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, leave=False, dynamic_ncols=True):
            # Move batch to device
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                batch = (images, labels)
            else:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            metrics = torch_validation_step(model, batch)
            average_meter.update(
                **{k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
            )

    metrics = average_meter.summary(prefix)
    num_samples = metrics.pop(prefix + "num_samples", 1)
    return {k: v / num_samples for k, v in metrics.items()}


###


def main():
    parser = argparse.ArgumentParser(description="Run PLSTM experiment")
    parser.add_argument("--config_path", type=str, default="./config", help="Path to config directory")
    parser.add_argument("--config_name", type=str, default="base", help="Name of base config file")
    parser.add_argument("--config_yaml", type=str, default="", help="Additional YAML config to override")
    parser.add_argument("--data_preloading_command", type=str, default="")
    parser.add_argument("--copy_model_checkpoint", type=str, default="")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--enable_trace", action="store_true")
    parser.add_argument("--check_reload_model", action="store_true")
    parser.add_argument("--disable_gc", action="store_true")
    parser.add_argument("--benchmark_dataloading", action="store_true")
    parser.add_argument("--dummy_data_test", action="store_true")
    parser.add_argument("--limit_loaders", type=int, default=None)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--just_print_config", action="store_true")
    parser.add_argument("--model_adaption", action="store_true")
    ### changed for torch
    # parser.add_argument("--use_nnx", action="store_true")
    parser.add_argument("--use_torch_compile", action="store_true", help="Use torch.compile for model optimization")
    ###
    parser.add_argument("--dist_info", type=str, default="")
    parser.add_argument("--multi_gpu_per_task", type=int, default=0)
    parser.add_argument(
        "opts",
        nargs="*",
        default=[],
        help="Additional arguments to override config (e.g. dataset.local_batch_size=32)",
    )

    args = parser.parse_args()
    if args.data_preloading_command:
        subprocess.run(args.data_preloading_command.split(" "))

    ### changed for torch
    # Initialize distributed training
    rank, world_size, local_rank, device = init_distributed()
    ###

    if args.enable_trace:
        import traceback
        import signal
        import threading

        def dump_trace(signum, frame):
            print("\n=== Stack trace ===")
            for thread_id, stack_frame in sys._current_frames().items():
                if thread_id == threading.main_thread().ident:
                    traceback.print_stack(stack_frame)

        signal.signal(signal.SIGUSR1, dump_trace)

    if "START_TIMESTAMP" in os.environ:
        extract_hydra.STARTTIME_STRING = os.environ["START_TIMESTAMP"]

    # Get config using hydra
    config_yaml = run_hydra(
        config_path=args.config_path,
        config_name=args.config_name,
        cmdline_opts=args.opts,
        config_yaml=args.config_yaml,
    )

    # Convert to ConfigDict for jax_trainer
    config_yaml_base = yaml.safe_load(config_yaml)
    config = OmegaConf.create(config_yaml_base)
    OmegaConf.resolve(config)
    ### changed for torch
    global_batch_size = config.dataset.local_batch_size * world_size
    ###
    assert global_batch_size == config.dataset.global_batch_size, (
        f"{config.dataset.local_batch_size} * {world_size} != {config.dataset.global_batch_size}"
    )
    LOGGER.info(f"GLOBAL BATCH SIZE: {config.dataset.global_batch_size}")

    config = OmegaConf.to_container(config)
    cfg_all = deepcopy(config)

    ### changed for torch
    cfg_all["slurm"] = {
        "devices": [str(device)],
        "processes": world_size,
        "local_devices": 1,
    }
    ###
    cfg_all["env"] = dict(**os.environ)
    cfg_all["config_unresolved"] = config_yaml
    cfg_all["config_args"] = sys.argv

    if args.print_config or args.just_print_config:
        print(yaml.safe_dump(config))
        if args.just_print_config:
            config = parse_config(ExperimentConfig, config)
            exit(0)

    config = parse_config(ExperimentConfig, config)
    np.random.seed(config.seed)
    random.seed(config.seed)
    ### changed for torch
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    ###

    dataset = config.dataset.instantiate(SimpleDatasetModule)

    exmp_input = next(iter(dataset.train_loader))

    if args.dummy_data_test:
        dataset.train_loader = itertools.cycle([exmp_input])
        dataset.val_loader = [exmp_input] * 8
        dataset.test_loader = [exmp_input] * 8

    if args.limit_loaders is not None:
        dataset.train_loader = LimitIterable(dataset.train_loader, args.limit_loaders)
        dataset.val_loader = LimitIterable(dataset.val_loader, args.limit_loaders)
        dataset.test_loader = LimitIterable(dataset.test_loader, args.limit_loaders)

    # Train model
    import cProfile
    import pstats
    from contextlib import nullcontext

    trainer_cfg = config.trainer

    print("LOGDIR: ", trainer_cfg.log_dir)
    os.makedirs(trainer_cfg.log_dir, exist_ok=True)

    ### changed for torch
    state = create_torch_train_state(
        trainer_cfg=config.trainer,
        model_cfg=config.model,
        optimizer_cfg=config.optimizer,
        device=device,
        use_torch_compile=args.use_torch_compile,
    )

    # Wrap model with DDP if distributed
    if world_size > 1 and dist.is_initialized():
        state.model = DDP(state.model, device_ids=[local_rank])
    ###

    ### changed for torch
    if trainer_cfg.logging and rank == 0:
        ###
        wandb.init(name=trainer_cfg.name, project=trainer_cfg.project, config=cfg_all, dir=trainer_cfg.log_dir)

    average_meter, max_val_acc1 = AverageMeter(use_latest=["learning_rate"]), 0.0
    train_loader_iter = iter(dataset.train_loader)

    assert trainer_cfg.train_batch_size == config.dataset.global_batch_size * trainer_cfg.grad_accum

    with cProfile.Profile() if args.profile else nullcontext() as pr:
        # main training here
        for step in tqdm.trange(1, trainer_cfg.train_steps + 1, dynamic_ncols=True):
            for accum_step in range(trainer_cfg.grad_accum):
                # Get next batch and move to device
                batch = next(train_loader_iter)
                ### changed for torch
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    batch = (images, labels)
                else:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                ###

                # Perform training step
                ### changed for torch
                metrics = torch_training_step(state, batch, accum_step, trainer_cfg.grad_accum)
                average_meter.update(
                    **{k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
                )
                ###

            # Log training metrics
            if trainer_cfg.log_interval > 0 and step % trainer_cfg.log_interval == 0:
                metrics = average_meter.summary(prefix="train/")
                metrics["processed_samples"] = step * trainer_cfg.train_batch_size
                ### changed for torch
                if trainer_cfg.logging and rank == 0:
                    ###
                    wandb.log(metrics, step)

            # Evaluate and save checkpoints
            if trainer_cfg.eval_interval > 0 and (
                step % trainer_cfg.eval_interval == 0 or step == trainer_cfg.train_steps
            ):
                ### changed for torch
                # Save checkpoint
                checkpoint = {
                    "model_state_dict": state.model.state_dict(),
                    "optimizer_state_dict": state.optimizer.state_dict(),
                    "step": step,
                    "config": config,
                }
                checkpoint_path = os.path.join(trainer_cfg.log_dir, f"{trainer_cfg.name}-last.pt")
                if rank == 0:
                    torch.save(checkpoint, checkpoint_path)
                ###

                if dataset.val_loader is None:
                    continue

                val_loaders = (
                    {"val": dataset.val_loader} if not isinstance(dataset.val_loader, dict) else dataset.val_loader
                )

                val_metrics = {}
                for val_idx, (val_prefix, val_loader) in enumerate(val_loaders.items()):
                    ### changed for torch
                    metrics = evaluate(
                        state.model, val_loader, device, prefix="val/" if val_idx == 0 else "val_" + val_prefix + "/"
                    )
                    ###

                    if val_idx == 0:
                        if metrics["val/acc1"] > max_val_acc1:
                            max_val_acc1 = metrics["val/acc1"]
                            ### changed for torch
                            best_checkpoint_path = os.path.join(trainer_cfg.log_dir, f"{trainer_cfg.name}-best.pt")
                            if rank == 0:
                                torch.save(checkpoint, best_checkpoint_path)
                            ###

                        metrics["val/acc1/best"] = max_val_acc1
                    val_metrics.update(**metrics)
                val_metrics["processed_samples"] = step * trainer_cfg.train_batch_size
                ### changed for torch
                if trainer_cfg.logging and rank == 0:
                    ###
                    wandb.log(val_metrics, step)

        if dataset.test_loader is not None:
            test_loaders = (
                {"test": dataset.test_loader} if not isinstance(dataset.test_loader, dict) else dataset.test_loader
            )

            test_metrics = {}
            for test_idx, (test_prefix, test_loader) in enumerate(test_loaders.items()):
                ### changed for torch
                metrics = evaluate(
                    state.model,
                    test_loader,
                    device,
                    prefix="test/" if test_idx == 0 else "test_" + test_prefix + "/",
                )
                ###
                test_metrics.update(**metrics)
            ### changed for torch
            if trainer_cfg.logging and rank == 0:
                ###
                wandb.log(test_metrics, step)
            val_metrics.update(**test_metrics)

    if args.profile:
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(100)

    # Print final metrics
    LOGGER.info(f"\nFinal validation metrics: {val_metrics['val/acc1']}")
    if dataset.test_loader is not None:
        LOGGER.info(f"\nFinal test metrics: {val_metrics['test/acc1']}")


if __name__ == "__main__":
    main()
