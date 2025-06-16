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
import jax
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
    TrainState,
    create_train_state,
    training_step,
    validation_step,
    create_train_nnx_state,
    training_nnx_step,
    validation_nnx_step,
)
from src.simple_dataset import SimpleDatasetModule
from src.utils import LimitIterable, AverageMeter, save_checkpoint_in_background

import src.simple_dataset  # noqa
import src.model_wrappers_linen  # noqa
import src.model_wrappers  # noqa
from jax_trainer.interfaces import BaseModelLinen, BaseModelNNX

from jax_trainer.optimizer import OptimizerInterface
import wandb
from torch.utils.data import DataLoader
from flax.jax_utils import unreplicate
from flax.serialization import msgpack_serialize
from flax.training.common_utils import shard
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from flax import nnx

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


def init_ddp_mesh(batch_mesh_axis: str = "data"):
    mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh((len(jax.devices()),)),
        (batch_mesh_axis,),
    )
    return mesh


@dataclass
class ExperimentConfig:
    trainer: VisionTrainerConfig
    model: BaseModelLinen.cfgtype | BaseModelNNX.cfgtype
    optimizer: OptimizerInterface.cfgtype
    dataset: SimpleDatasetModule.cfgtype
    batch_mesh_axis: Any
    slurm: Any = None
    env: Any = None
    config_unresolved: Any = None
    config_args: Any = None
    seed: int = 42
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(unsafe_hash=True)
class MeshRules:
    data: str | None = None

    def __call__(self, *keys: str) -> tuple[str]:
        return tuple(getattr(self, key) for key in keys)


def evaluate(
    state: TrainState | Any, dataloader: DataLoader, use_nnx: bool = False, prefix="val/", mesh: Mesh = None
) -> dict[str, float]:
    average_meter = AverageMeter()
    for batch in tqdm.tqdm(dataloader, leave=False, dynamic_ncols=True):
        # Convert PyTorch tensors to NumPy arrays, shard across devices
        if use_nnx:
            batch = jax.device_put(
                jax.tree_util.tree_map(lambda x: np.asarray(x.numpy() if hasattr(x, "numpy") else x), batch),
                NamedSharding(mesh, P("data")),
            )
            metrics = validation_nnx_step(state, batch)
        else:
            batch = shard(jax.tree_util.tree_map(lambda x: np.asarray(x.numpy() if hasattr(x, "numpy") else x), batch))
            metrics = validation_step(state, batch)
        average_meter.update(**jax.device_get(unreplicate(metrics)))

    metrics = average_meter.summary(prefix)
    num_samples = metrics.pop(prefix + "num_samples")
    return jax.tree_util.tree_map(lambda x: x / num_samples, metrics)


def main():
    parser = argparse.ArgumentParser(description="Run PLSTM experiment")
    parser.add_argument("--config_path", type=str, default="./config", help="Path to config directory")
    parser.add_argument("--config_name", type=str, default="base", help="Name of base config file")
    parser.add_argument("--config_yaml", type=str, default="", help="Additional YAML config to override")
    parser.add_argument("--data_preloading_command", type=str, default="")
    parser.add_argument("--copy_model_checkpoint", type=str, default="")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--enable_trace", action="store_true")
    # parser.add_argument("--new_data_lib", action="store_true")
    parser.add_argument("--check_reload_model", action="store_true")
    parser.add_argument("--disable_gc", action="store_true")
    parser.add_argument("--benchmark_dataloading", action="store_true")
    parser.add_argument("--dummy_data_test", action="store_true")
    parser.add_argument("--limit_loaders", type=int, default=None)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--just_print_config", action="store_true")
    parser.add_argument("--model_adaption", action="store_true")
    parser.add_argument("--use_nnx", action="store_true")
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

    process_id, num_processes = 0, 1
    if (
        "SLURM_NTASKS" in os.environ and int(os.environ["SLURM_NTASKS"]) > 1 and "SLURM_STEP_NODELIST" in os.environ
    ) or args.dist_info:
        # Initializes one process per device, using the SLURM environment variables.
        # TODO: We may need to do this already before data loading, so very early in the run script.
        # To be checked once the framework is more mature.
        # This should also NOT be called for tests, as it was called already before the test.
        # If you run a test, unset this SLURM variable as a workaround.
        # main_compilation_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", ".")
        # os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.join(
        #     main_compilation_cache_dir, "proc_" + os.environ["SLURM_PROCID"]
        # )
        # print("JAX_COMPILATION_CACHE_DIR:", os.environ["JAX_COMPILATION_CACHE_DIR"])
        # os.makedirs(os.environ["JAX_COMPILATION_CACHE_DIR"], exist_ok=True)
        # jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
        # multiprocessing.set_start_method("spawn", force=True)
        if args.dist_info:
            dist_info = args.dist_info.split(",")
            if len(dist_info) != 3:
                raise ValueError("Bad distinfo format, expected COORDINATOR_ADDR:PORT,PROC_ID,N_PROCS")
            coordinator_address, process_id, num_processes = dist_info
            num_processes = int(num_processes)
            process_id = int(process_id)
            jax.distributed.initialize(
                coordinator_address=coordinator_address, num_processes=num_processes, process_id=process_id
            )
        else:
            if args.multi_gpu_per_task:
                jax.distributed.initialize(local_device_ids=range(args.multi_gpu_per_task))
            else:
                jax.distributed.initialize()
        mesh: Mesh = init_ddp_mesh()
    else:
        mesh: Mesh = init_ddp_mesh()

    def named_sharding(*names: str | None) -> NamedSharding:
        # P(*names) creates a PartitionSpec, e.g., P('data', None)
        # NamedSharding binds this PartitionSpec to the 'mesh'.
        return NamedSharding(mesh, P(*names))

    LOGGER.info(
        f"JAX DEVICES: {jax.devices()}, local devices: {jax.local_devices()}, proc_id: {jax.process_index()}, nproc: {jax.process_count()}, Mesh: {mesh}"
        f", CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    # LOGGER.info(f"JAX DEVICES: {jax.devices()}")

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
    global_batch_size = config.dataset.local_batch_size * len(jax.devices())
    assert global_batch_size == config.dataset.global_batch_size, (
        f"{config.dataset.local_batch_size} * {len(jax.devices())} != {config.dataset.global_batch_size}"
    )
    LOGGER.info(f"GLOBAL BATCH SIZE: {config.dataset.global_batch_size}")

    config = OmegaConf.to_container(config)
    cfg_all = deepcopy(config)

    cfg_all["slurm"] = {
        "devices": [str(i) for i in jax.devices()],
        "processes": int(jax.process_count()),
        "local_devices": int(jax.local_device_count()),
        # "mesh": str(mesh),
    }
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
    if args.use_nnx:
        state = create_train_nnx_state(
            trainer_cfg=config.trainer,
            model_cfg=config.model,
            optimizer_cfg=config.optimizer,
            mesh=mesh,
        )

    else:
        state = create_train_state(
            trainer_cfg=config.trainer,
            model_cfg=config.model,
            optimizer_cfg=config.optimizer,
        ).replicate()

    if trainer_cfg.logging and jax.process_index() == 0:
        wandb.init(name=trainer_cfg.name, project=trainer_cfg.project, config=cfg_all, dir=trainer_cfg.log_dir)

    average_meter, max_val_acc1 = AverageMeter(use_latest=["learning_rate"]), 0.0
    train_loader_iter = iter(dataset.train_loader)

    assert trainer_cfg.train_batch_size == config.dataset.global_batch_size

    with cProfile.Profile() if args.profile else nullcontext() as pr:
        # main training here
        for step in tqdm.trange(1, trainer_cfg.train_steps + 1, dynamic_ncols=True):
            for _ in range(trainer_cfg.grad_accum):
                # Get next batch, convert to NumPy arrays, and shard across devices
                batch = next(train_loader_iter)
                if args.use_nnx:
                    batch = jax.device_put(
                        jax.tree_util.tree_map(lambda x: np.asarray(x.numpy() if hasattr(x, "numpy") else x), batch),
                        named_sharding("data"),
                    )
                else:
                    batch = shard(
                        jax.tree_util.tree_map(lambda x: np.asarray(x.numpy() if hasattr(x, "numpy") else x), batch)
                    )

                # Perform training step
                if args.use_nnx:
                    metrics = training_nnx_step(state, batch)
                else:
                    state, metrics = training_step(state, batch)

                average_meter.update(**jax.device_get(unreplicate(metrics)))

            # Log training metrics
            if trainer_cfg.log_interval > 0 and step % trainer_cfg.log_interval == 0:
                metrics = average_meter.summary(prefix="train/")
                metrics["processed_samples"] = step * trainer_cfg.train_batch_size
                if trainer_cfg.logging and jax.process_index() == 0:
                    wandb.log(metrics, step)

            # Evaluate and save checkpoints
            if trainer_cfg.eval_interval > 0 and (
                step % trainer_cfg.eval_interval == 0 or step == trainer_cfg.train_steps
            ):
                if args.use_nnx:
                    params_bytes = msgpack_serialize(nnx.state(state, nnx.Params))
                else:
                    params_bytes = msgpack_serialize(unreplicate(state.params))
                save_checkpoint_in_background(config, params_bytes, postfix="last")

                if dataset.val_loader is None:
                    continue

                val_loaders = (
                    {"val": dataset.val_loader} if not isinstance(dataset.val_loader, dict) else dataset.val_loader
                )

                val_metrics = {}
                for val_idx, (val_prefix, val_loader) in enumerate(val_loaders.items()):
                    metrics = evaluate(state, val_loader, prefix="val/" if val_idx == 0 else "val_" + val_prefix + "/")

                    if val_idx == 0:
                        if metrics["val/acc1"] > max_val_acc1:
                            max_val_acc1 = metrics["val/acc1"]
                            save_checkpoint_in_background(config, params_bytes, postfix="best")

                        metrics["val/acc1/best"] = max_val_acc1
                    val_metrics.update(**metrics)
                val_metrics["processed_samples"] = step * trainer_cfg.train_batch_size
                if trainer_cfg.logging and jax.process_index() == 0:
                    wandb.log(val_metrics, step)

        if dataset.test_loader is not None:
            test_loaders = (
                {"test": dataset.test_loader} if not isinstance(dataset.test_loader, dict) else dataset.test_loader
            )

            test_metrics = {}
            for test_idx, (test_prefix, test_loader) in enumerate(test_loaders.items()):
                metrics = evaluate(
                    state,
                    test_loader,
                    use_nnx=args.use_nnx,
                    prefix="test/" if test_idx == 0 else "test_" + test_prefix + "/",
                    mesh=mesh,
                )
                test_metrics.update(**metrics)
            if trainer_cfg.logging and jax.process_index() == 0:
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
