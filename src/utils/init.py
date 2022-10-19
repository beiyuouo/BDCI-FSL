import sys
import os
import yaml
from loguru import logger
from pathlib import Path
import ezkfg as ez


def init_log(log_path: str = None):
    """Initialize loguru log information"""
    event_logger_format = (
        "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | "
        "<lvl>{level}</lvl> - "
        # "<c><u>{name}</u></c> | "
        "{message}"
    )
    logger.remove()
    logger.add(
        sink=sys.stdout,
        colorize=True,
        level="DEBUG",
        format=event_logger_format,
        diagnose=False,
    )

    if log_path is not None:
        logger.add(
            sink=log_path / "log.txt",
            colorize=False,
            level="DEBUG",
            format=event_logger_format,
            diagnose=False,
        )

    return logger


def init_seed(seed: int = 3407):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_path(cfg):
    cfg["run_path"] = (
        Path(cfg["run_path"])
        / f"{cfg['exp_name']}_ep{cfg['epochs']}_bs{cfg['batch_size']}_warmup{cfg['num_warmup_steps']}_lr{cfg['encoder_lr']}"
    )
    cfg["run_path"].mkdir(parents=True, exist_ok=True)
    cfg["log_path"] = cfg["run_path"] / "logs"
    cfg["log_path"].mkdir(parents=True, exist_ok=True)
    cfg["model_path"] = cfg["run_path"] / "weights"
    cfg["model_path"].mkdir(parents=True, exist_ok=True)
    cfg["export_path"] = Path(cfg["export_path"])
    cfg["export_path"].mkdir(parents=True, exist_ok=True)


def init(cfg_path: str, call_from: str = None):
    cfg = ez.Config().load(cfg_path)

    init_seed(cfg.seed)
    init_path(cfg)
    init_log(cfg.log_path)

    # save to run_path
    yaml.dump(cfg, open(cfg["run_path"] / "hyps.yaml", "w"))
    return cfg
