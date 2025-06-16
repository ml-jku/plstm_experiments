#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Any

# from ml_collections import ConfigDict
from jax_trainer.callbacks import ModelCheckpoint
import multiprocessing
import shutil
from jax.experimental import multihost_utils
import shutil
import subprocess
from tqdm import tqdm
from collections.abc import Iterable

from omegaconf import OmegaConf
import jax
from flax import nnx

import time
import os
import numpy as np

import plstm.nnx  # noqa
from jax_trainer.init_mesh import init_ddp_mesh

from compoconf import parse_config
import sys
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent))
from src import model_wrappers, model_wrappers_linen  # noqa , used to register basic models
from src import extract_hydra
from src.extract_hydra import run_hydra

# from jax_trainer.datasets import build_dataset_module as build_dataset_module_old
# from src.datasets import build_dataset_module
from src.custom_datasets import *  # noqa , used to register datasets
import mill  # noqa  used to register preprocessing and augmentation modules
from jax_trainer.interfaces import BaseModelNNX  # noqa
from jax_trainer.datasets import DatasetModule
from jax_trainer.optimizer import OptimizerInterface
from jax_trainer.trainer import TrainerModule

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


class TqdmWithStddev(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speeds = []

    def update(self, n=1):
        if self.last_print_t and self.n:
            speed = n / (time.time() - self.last_print_t)
            self.speeds.append(speed)
            if len(self.speeds) > 100:  # Limit number of stored speeds
                self.speeds.pop(0)
        super().update(n)

    def format_dict(self):
        d = super().format_dict()
        if self.speeds:
            d["stddev"] = np.std(self.speeds)
        else:
            d["stddev"] = 0.0
        return d

    def format_meter(self, *args, **kwargs):
        meter = super().format_meter(*args, **kwargs)
        if "stddev" in self.format_dict():
            meter += f" ± {self.format_dict()['stddev']:.2f} it/s"
        return meter


class LimitIterable:
    def __init__(self, it: Iterable, limit: int = None):
        self.it = it
        self._it = None
        self.limit = limit
        self.count = 0

    def __len__(self):
        real_len = len(self.it)
        return min(real_len, self.limit) if self.limit is not None else real_len

    def __iter__(self):
        self._it = iter(self.it)
        self.count = 0
        return self

    def __next__(self):
        self.count += 1
        if self.limit is not None and self.count == self.limit:
            raise StopIteration
        return next(self._it)


def adapt_model(
    model,
    resolution,
    patch_size,
):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if hasattr(model, "pos_embed"):
        model.pos_embed.embed = nnx.Param(
            jax.image.resize(
                model.pos_embed.embed,
                (1, resolution[0] // patch_size[0], resolution[1] // patch_size[1], model.pos_embed.embed.shape[-1]),
                method="bicubic",
            )
        )


@dataclass
class ExperimentConfig:
    trainer: TrainerModule.cfgtype
    model: BaseModelNNX.cfgtype
    optimizer: OptimizerInterface.cfgtype
    dataset: DatasetModule.cfgtype
    batch_mesh_axis: Any
    slurm: Any = None
    env: Any = None
    config_unresolved: Any = None
    config_args: Any = None
    seed: int = 42
    aux: dict[str, Any] = field(default_factory=dict)


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
    parser.add_argument("--limit_dataloaders", type=int, default=None)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--just_print_config", action="store_true")
    parser.add_argument("--model_adaption", action="store_true")
    parser.add_argument("--initial_save", action="store_true")
    parser.add_argument("--dist_info", type=str, default="")

    parser.add_argument(
        "opts",
        nargs="*",
        default=[],
        help="Additional arguments to override config (e.g. dataset.local_batch_size=32)",
    )
    print("CUDA_VISIBLE_DEVICES: ", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

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
            jax.distributed.initialize()
        mesh = init_ddp_mesh()
    else:
        mesh = init_ddp_mesh()

    LOGGER.info(
        f"JAX DEVICES: {jax.devices()}, proc_id: {jax.process_index()}, nproc: {jax.process_count()}, Mesh: {mesh}"
    )

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

    # print(config)
    config = OmegaConf.to_container(config)
    cfg_other = deepcopy(config)
    del cfg_other["trainer"]
    del cfg_other["model"]
    del cfg_other["optimizer"]
    cfg_other["slurm"] = {
        "devices": [str(i) for i in jax.devices()],
        "processes": int(jax.process_count()),
        "local_devices": int(jax.local_device_count()),
        "mesh": str(mesh),
    }
    cfg_other["env"] = list(os.environ)
    cfg_other["config_unresolved"] = config_yaml
    cfg_other["config_args"] = sys.argv

    # config = ConfigDict(config)

    if args.print_config or args.just_print_config:
        print(yaml.safe_dump(config))
        if args.just_print_config:
            config = parse_config(ExperimentConfig, config)
            exit(0)

    config = parse_config(ExperimentConfig, config)

    # parse_config(BaseModelNNX.cfgtype | model_wrappers.ViTConfig, config.model)
    # dataset_cfg = parse_config(DatasetModule.cfgtype, config["dataset"])

    # Build dataset
    # if args.new_data_lib:
    # dataset = build_dataset_module(config.dataset, mesh=mesh, rng_key=jax.random.PRNGKey(config.dataset.seed))
    # else:
    #     dataset = build_dataset_module_old(config.dataset, mesh=mesh)
    dataset = config.dataset.instantiate(DatasetModule, mesh=mesh)

    exmp_input = next(iter(dataset.train_loader))
    if args.benchmark_dataloading:
        if args.disable_gc:
            import gc

            gc.disable()

        for batch in tqdm(
            dataset.train_loader,
            total=len(dataset.train_loader),
        ):
            _ = batch
        if args.disable_gc:
            import gc

            gc.enable()

    if args.dummy_data_test:
        dataset.train_loader = [exmp_input] * len(dataset.train_loader)
        dataset.val_loader = [exmp_input] * len(dataset.val_loader)
        dataset.test_loader = [exmp_input] * len(dataset.test_loader)

    if args.limit_dataloaders is not None:
        dataset.train_loader = LimitIterable(dataset.train_loader, args.limit_dataloaders)
        dataset.val_loader = LimitIterable(dataset.val_loader, args.limit_dataloaders)
        dataset.test_loader = LimitIterable(dataset.test_loader, args.limit_dataloaders)

    # Initialize trainer
    trainer = config.trainer.instantiate(
        TrainerModule,
        model_config=config.model,
        optimizer_config=config.optimizer,
        exmp_input=exmp_input,
        other_configs=cfg_other,
        mesh=mesh,
    )

    LOGGER.info(f"LOGDIR: {trainer.logger.log_dir}")

    LOGGER.info(f"CALLBACKS: {trainer.callbacks}")
    if args.copy_model_checkpoint:
        if jax.process_index() == 0:
            shutil.copytree(args.copy_model_checkpoint, Path(trainer.log_dir) / "checkpoints" / "checkpoint_0")

        multihost_utils.sync_global_devices("barrier")
        LOGGER.info(f"Checkpoint from {args.copy_model_checkpoint} copied.")
        trainer.load_model()
        LOGGER.info("Transferred Checkpoint loaded")
        if args.model_adaption:
            adapt_model(trainer.model.wrapped_model, config.dataset.resolution, config.model.patch_size)

        multihost_utils.sync_global_devices("barrier")
        if jax.process_index() == 0:
            shutil.rmtree(Path(trainer.log_dir) / "checkpoints" / "checkpoint_0")
        LOGGER.info("Transferred Checkpoint removed")
        multihost_utils.sync_global_devices("barrier")

    if args.initial_save:
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.save_model({"val/acc": 0}, epoch_idx=0)

                LOGGER.info("MODEL SAVED")

                n = 0
                while not (Path(callback.manager.directory) / "checkpoint_0").exists() and n <= 20:
                    LOGGER.info("Wait for model to be saved.")
                    time.sleep(5)
                    n += 5
                break

            if args.check_reload_model:
                trainer.load_model()

    if args.disable_gc:
        import gc

        gc.disable()

    # Train model
    if args.profile:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            eval_metrics = trainer.train_model(
                train_loader=dataset.train_loader,
                val_loader=dataset.val_loader,
                test_loader=dataset.test_loader,
                num_epochs=trainer.config.train_epochs,
            )
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(100)
    else:
        eval_metrics = trainer.train_model(
            train_loader=dataset.train_loader,
            val_loader=dataset.val_loader,
            test_loader=dataset.test_loader,
            num_epochs=trainer.config.train_epochs,
        )

    # Print final metrics
    LOGGER.info(f"\nFinal validation metrics: {eval_metrics[trainer.config.train_epochs]}")
    LOGGER.info(f"Test metrics: {eval_metrics['test']}")


if __name__ == "__main__":
    main()
