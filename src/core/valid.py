import os
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.init import init
from utils.data import load_data
from core.model import load_model, load_tokenizer
from core.dataset import FSDataset
from core.train import valid_epoch

def valid(cfg_path: str, model_path: str):
    cfg = init(cfg_path)
    logger.info(cfg)

    if model_path is None:
        model_path = (
            cfg.model_path / "best" if cfg.use_best else cfg.model_path / "final"
        )

    model = load_model(model_path)
    logger.info(model)

    tokenizer = load_tokenizer(model_path)
    data_df = load_data(cfg.data_path, split="train")
    logger.info(data_df.head())
    train_data = data_df.sample(frac=0.8, random_state=cfg.seed)
    valid_data = data_df.drop(train_data.index).reset_index(drop=True)
    assert(set(valid_data["label_id"])==set(range(36)))
    valid_dataset = FSDataset(valid_data, tokenizer)
    valid_dl = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.to(cfg.device)
    model.eval()

    valid_epoch(0,model, valid_dl, cfg.device, cfg,verbose=True)

