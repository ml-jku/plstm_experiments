from hydra import initialize_config_dir, compose
import os
from typing import List, Dict, Union
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from math import sqrt as _sqrt
from datetime import datetime

STARTTIME_STRING = ""


def safe_mul(*args):
    res = 1
    for arg in args:
        try:
            arg = float(arg)
            res *= arg
        except ValueError:
            pass
    return res


def safe_muli(*args):
    res = 1
    for arg in args:
        try:
            arg = int(arg)
            res *= arg
        except ValueError:
            pass
    return res


def sqrt(*args):
    return float(_sqrt(float(args[0])))


def oc_slice(*args):
    return args[0][int(args[1]) : int(args[2])]


def oc_floor_divide(*args):
    if isinstance(args[0], ListConfig):
        if isinstance(args[1], ListConfig):
            return ListConfig([arg // arg1 for arg, arg1 in zip(args[0], args[1])])
        else:
            return ListConfig([arg // args[1] for arg in args[0]])
    else:
        return args[0] // args[1]


def oc_ceil_divide(cfg1, cfg2):
    return (int(cfg1) - 1) // int(cfg2) + 1


def oc_mul_round_int(*args):
    # computes a * b rounded to a multiple of c
    return int(round(args[0] * args[1] / args[2]) * args[2])


def oc_timestring(*args):
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3] if not STARTTIME_STRING else STARTTIME_STRING


def oc_merge(cfg1, cfg2):
    return OmegaConf.merge(cfg1, cfg2)


def oc_concat(cfg1, cfg2):
    return cfg1 + cfg2


def oc_subi(cfg1, cfg2):
    return int(cfg1) - int(cfg2)


def oc_addi(cfg1, cfg2):
    return int(cfg1) + int(cfg2)


OmegaConf.register_new_resolver("oc.mul", safe_mul, replace=True)
OmegaConf.register_new_resolver("oc.muli", safe_muli, replace=True)
OmegaConf.register_new_resolver("oc.subi", oc_subi, replace=True)
OmegaConf.register_new_resolver("oc.addi", oc_addi, replace=True)
OmegaConf.register_new_resolver("oc.sqrt", sqrt, replace=True)
OmegaConf.register_new_resolver("oc.len", len, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("oc.slice", oc_slice, replace=True)
OmegaConf.register_new_resolver("oc.floor_divide", oc_floor_divide, replace=True)
OmegaConf.register_new_resolver("oc.ceil_divide", oc_ceil_divide, replace=True)

OmegaConf.register_new_resolver("oc.mul_round_int", oc_mul_round_int, replace=True)
OmegaConf.register_new_resolver("oc.timestring", oc_timestring, replace=True)
OmegaConf.register_new_resolver("oc.merge", oc_merge, use_cache=False)
OmegaConf.register_new_resolver("oc.concat", oc_concat, use_cache=False)


def config_yaml_to_cmdline(config_yaml: str, override: str = "") -> List[str]:
    # override either "", "+", "++" see hydra
    cfg = OmegaConf.create(config_yaml)
    cfg_dict = OmegaConf.to_container(cfg)
    cmdline_opts = []

    def dict_to_cmdlines(dct: Union[Dict, List, str, int, float], prefix: str = ""):
        cmdlines = []

        if isinstance(dct, Dict):
            for sub_cfg in dct:
                newprefix = (prefix + "." if prefix else "") + sub_cfg
                cmdlines += dict_to_cmdlines(dct[sub_cfg], prefix=newprefix)
        elif isinstance(dct, List):
            cmdlines.append(override + prefix + "=[" + ",".join(map(str, range(len(dct)))) + "]")
            for n, sub_cfg in enumerate(dct):
                cmdlines += dict_to_cmdlines(sub_cfg, prefix=(prefix + "." if prefix else "") + str(n))
        elif dct is None:
            cmdlines.append(override + prefix + "=null")
        else:
            cmdlines.append(override + prefix + "=" + str(dct))
        return cmdlines

    # old version
    # for sub_cfg in cfg_dict:
    #     print(sub_cfg, cfg_dict[sub_cfg])
    #     cfg_json = json.dumps(cfg_dict[sub_cfg], separators=(',', ':'))
    #     cmdline_opts.append(override + sub_cfg + "=" + cfg_json)
    cmdline_opts = dict_to_cmdlines(cfg_dict, prefix="")
    # print(cmdline_opts)
    return cmdline_opts


def run_hydra(
    config_path: str = "./config",
    config_name: str = "default",
    cmdline_opts=[],
    config_yaml: str = "",
    config_yaml_override_opt: str = "++",
):
    # do not actually run hydra as a separate executable

    config_path = config_path if os.path.isabs(config_path) else os.path.abspath(config_path)
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(
            config_name=config_name,
            overrides=config_yaml_to_cmdline(config_yaml, override=config_yaml_override_opt) + cmdline_opts,
        )

    return OmegaConf.to_yaml(cfg)
