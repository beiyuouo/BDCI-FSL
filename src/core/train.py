from operator import mod
import os
import time
from pathlib import Path
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import numpy as np
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
from core.loss import get_criterion


def train_epoch(
    epoch, model, data_loader, criterion, optimizer, scheduler, device, cfg, writer=None
):
    model.train()
    model.to(device)
    total_loss = 0.0

    for step, batch in enumerate(data_loader):
        label = batch["label"].to(device)
        mask = batch["attention_mask"].to(device)
        input_ids = batch["input_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=mask,
        )

        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm
        )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if (step + 1) % cfg.log_freq == 0 or step == len(data_loader) - 1:
            logger.info(
                f"epoch {epoch + 1} / {cfg.epochs} step {step + 1} / {len(data_loader)} loss {loss.item():.4f} loss avg {total_loss / (step + 1):.4f} grad norm {grad_norm:.4f}"
            )

    logger.info(
        f"epoch {epoch + 1} / {cfg.epochs} train loss {total_loss / (step + 1):.4f}"
    )


def valid_epoch(model, data_loader, criterion, device, cfg, writer=None):
    model.eval()
    model.to(device)
    total_loss = 0.0
    preds = []
    labels = []

    for step, batch in enumerate(data_loader):
        label = batch["label"].to(device)
        mask = batch["attention_mask"].to(device)
        input_ids = batch["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=mask,
            )

        loss = criterion(outputs, label)

        total_loss += loss.item()

        preds.extend(outputs.argmax(dim=1).to("cpu").numpy())
        labels.extend(label.to("cpu").numpy())

        if (step + 1) % cfg.log_freq == 0 or step == len(data_loader) - 1:
            logger.info(
                f"step {step + 1} / {len(data_loader)} loss {loss.item():.4f} loss avg {total_loss / (step + 1):.4f}"
            )

    preds = np.array(preds)
    labels = np.array(labels)
    f1 = f1_score(labels, preds, average="macro")
    # logger.info(f"f1 score {f1:.4f}")

    return f1


def train_(cfg, model, tokenizer, train_df, valid_df, device, writer=None):
    assert (
        len(train_df["label_id"].unique()) == cfg.num_labels
    ), "train data labels do not have all labels"
    assert (
        len(valid_df["label_id"].unique()) == cfg.num_labels
    ), "valid data labels do not have all labels"

    train_dt = FSDataset(train_df, tokenizer, cfg=cfg)
    valid_dt = FSDataset(valid_df, tokenizer, cfg=cfg)

    train_dl = DataLoader(
        train_dt,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    valid_dl = DataLoader(
        valid_dt,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    criterion = get_criterion(cfg)
    optimizer = get_optimizer(model, cfg)
    num_train_steps = int(len(train_dt) / cfg.batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    best_f1 = 0
    last_f1 = 0

    for epoch in range(cfg.epochs):
        logger.info(f"epoch {epoch + 1} / {cfg.epochs}")
        start_time = time.time()

        train_epoch(
            epoch, model, train_dl, criterion, optimizer, scheduler, device, cfg, writer
        )
        logger.info(
            f"train epoch {epoch + 1} / {cfg.epochs} done in {time.time() - start_time:.2f} seconds"
        )

        start_time = time.time()
        f1_train = valid_epoch(model, train_dl, criterion, device, cfg, writer)
        f1_val = valid_epoch(model, valid_dl, criterion, device, cfg, writer)
        logger.info(
            f"valid epoch {epoch + 1} / {cfg.epochs} done in {time.time() - start_time:.2f} seconds f1 train {f1_train:.4f} f1 val {f1_val:.4f}"
        )

        if f1_val > best_f1:
            best_f1 = f1_val
            logger.info(f"best fl val score {best_f1:.4f} saving model")
            tokenizer.save_pretrained(cfg.model_path / "best")
            torch.save(model.state_dict(), cfg.model_path / "best" / "model.pth")

        if (epoch + 1) % cfg.chpt_freq == 0:
            logger.info(f"saving model at epoch {epoch + 1}")
            tokenizer.save_pretrained(cfg.model_path / f"chpt_{epoch + 1}")
            torch.save(
                model.state_dict(), cfg.model_path / f"chpt_{epoch + 1}" / "model.pth"
            )

        last_f1 = f1_val

    logger.info(f"best f1 score {best_f1:.4f}")
    tokenizer.save_pretrained(cfg.model_path / "final")
    torch.save(model.state_dict(), cfg.model_path / "final" / "model.pth")

    return best_f1, last_f1


def train(cfg_path: str, model_path: str = None):
    cfg = init(cfg_path)
    logger.info(cfg)

    data_df = load_data(cfg.data_path, split="train")
    logger.info(data_df.head())

    if cfg.use_kfold:
        # add kfold column
        for i in range(cfg.num_labels):
            # add fold for each label with same distribution
            len_label = len(data_df[data_df["label_id"] == i])
            num_folds = cfg.num_folds
            if len_label < cfg.num_folds:
                num_folds = len_label

            data_df.loc[data_df["label_id"] == i, "fold"] = (
                np.arange(len_label) % cfg.num_folds
            )

        data_df["fold"] = data_df["fold"].astype(int)
        # data_df[data_df["label"] == i]["fold"] = np

    else:
        train_df = data_df.sample(frac=0.8, random_state=cfg.seed)
        valid_df = data_df.drop(train_df.index).reset_index(drop=True)

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

    if cfg.use_tensorboard:
        tb_writer = SummaryWriter(cfg.log_path)
    else:
        tb_writer = None

    if cfg.use_kfold:
        best_f1, last_f1 = [], []
        for fold in range(cfg.num_folds):
            logger.info(f"fold {fold + 1} / {cfg.num_folds}")
            train_df = data_df[data_df["fold"] != fold]
            valid_df = data_df[data_df["fold"] == fold]

            if cfg.from_scratch:
                logger.info("training from scratch")
                model = get_model(cfg.model_name, cfg.num_labels)
            else:
                logger.info(f"loading model from {model_path}")
                model = load_model(model_path, cfg.num_labels)

            best_f1_, last_f1_ = train_(
                cfg, model, tokenizer, train_df, valid_df, cfg.device, tb_writer
            )
            best_f1.append(best_f1_)
            last_f1.append(last_f1_)

            logger.info(f"fold {fold + 1} / {cfg.num_folds} done")

        logger.info(
            f"best f1 score {np.mean(best_f1):.4f} last f1 score {np.mean(last_f1):.4f}"
        )
    else:
        train_(cfg, model, tokenizer, train_df, valid_df, cfg.device, tb_writer)
