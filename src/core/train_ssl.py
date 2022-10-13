import os
import time
from pathlib import Path
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import ezkfg as ez

from utils.init import init
from utils.data import load_data
from core.dataset import FSDataset
from core.model import (
    get_model,
    get_optimizer,
    get_scheduler,
    get_tokenizer,
    load_model,
)
from core.train import train_epoch, valid_epoch


def get_preds_with_probs(model, loader, device):
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def train(cfg_path: str, model_path: str = None):
    cfg = init(cfg_path)
    logger.info(cfg)

    data_df = load_data(cfg.data_path, split="train")
    logger.info(data_df.head())

    train_data = data_df.sample(frac=0.9, random_state=cfg.seed)
    valid_data = data_df.drop(train_data.index).reset_index(drop=True)
    unlabel_data = load_data(cfg.data_path, split="test")

    model_path = (
        Path(model_path) if model_path is not None else cfg.model_path / "pretrained"
    )

    if cfg.from_scratch:
        logger.info("training from scratch")
        model = get_model(cfg.model_name, cfg.num_labels)
    else:
        logger.info(f"loading model from {model_path}")
        model = load_model(model_path, cfg.num_labels)

    tokenizer = get_tokenizer(cfg.model_name)

    train_dataset = FSDataset(train_data, tokenizer)
    valid_dataset = FSDataset(valid_data, tokenizer)
    unlabel_dataset = FSDataset(unlabel_data, tokenizer, is_test=True)

    train_dl = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    valid_dl = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    unlabel_dl = DataLoader(
        unlabel_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model.to(cfg.device)

    optimizer = get_optimizer(model, cfg)
    num_train_steps = int(len(train_dataset) / cfg.batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    best_f1 = 0

    for epoch in range(cfg.epochs):
        logger.info(f"epoch {epoch + 1} / {cfg.epochs}")
        start_time = time.time()

        train_epoch(epoch, model, train_dl, optimizer, scheduler, cfg.device, cfg)
        logger.info(
            f"train epoch {epoch + 1} / {cfg.epochs} done in {time.time() - start_time:.2f} seconds"
        )

        if epoch >= cfg.warmup_epochs:
            unlabel_preds, unlabel_probs = get_preds_with_probs(
                model, unlabel_dl, cfg.device
            )
            unlabel_data["label_id"] = unlabel_preds
            unlabel_data["probs"] = unlabel_probs.max(axis=1)
            labeled_data = unlabel_data[unlabel_data["probs"] > cfg.threshold]
            unlabel_data = unlabel_data[unlabel_data["probs"] <= cfg.threshold]
            if len(labeled_data) > 0:
                train_data = pd.concat([train_data, labeled_data], axis=0).reset_index(
                    drop=True
                )
                train_dataset = FSDataset(train_data, tokenizer)
                train_dl = DataLoader(
                    train_dataset,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                )

                unlabel_dataset = FSDataset(unlabel_data, tokenizer)
                unlabel_dl = DataLoader(
                    unlabel_dataset,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                )

            logger.info(f"add {len(labeled_data)} samples to train data")

        start_time = time.time()
        f1_train = valid_epoch(model, train_dl, cfg.device, cfg)
        f1_val = valid_epoch(model, valid_dl, cfg.device, cfg)
        logger.info(
            f"valid epoch {epoch + 1} / {cfg.epochs} done in {time.time() - start_time:.2f} seconds f1 train {f1_train:.4f} f1 val {f1_val:.4f}"
        )

        if f1_val > best_f1:
            best_f1 = f1_val
            logger.info(f"best fl val score {best_f1:.4f} saving model")
            model.save_pretrained(cfg.model_path / "best")
            tokenizer.save_pretrained(cfg.model_path / "best")

        if (epoch + 1) % cfg.chpt_freq == 0:
            logger.info(f"saving model at epoch {epoch + 1}")
            model.save_pretrained(cfg.model_path / f"chpt_{epoch + 1}")
            tokenizer.save_pretrained(cfg.model_path / f"chpt_{epoch + 1}")

    logger.info(f"best f1 score {best_f1:.4f}")
    model.save_pretrained(cfg.model_path / "final")
    tokenizer.save_pretrained(cfg.model_path / "final")
