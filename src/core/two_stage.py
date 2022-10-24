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
from utils.data import load_data, get_two_stage_data
from core.dataset import FSDataset
from core.model import (
    get_model,
    get_optimizer,
    get_scheduler,
    get_tokenizer,
    load_model,
    load_tokenizer,
)
from core.loss import get_criterion


def train_epoch(
    epoch,
    model,
    data_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    cfg,
    writer=None,
    _prefix="",
):
    model.train()
    model.to(device)
    total_loss = 0.0
    preds = []
    labels = []

    for step, batch in enumerate(data_loader):
        label = batch["label"].to(device)
        mask = batch["attention_mask"].to(device)
        input_ids = batch["input_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=mask,
        )

        preds.extend(outputs.argmax(dim=1).to("cpu").numpy())
        labels.extend(label.to("cpu").numpy())

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

        writer.add_scalars(
            f"train_step/loss", {_prefix: loss.item()}, epoch * len(data_loader) + step
        )
        writer.add_scalars(
            f"train_step/grad_norm",
            {_prefix: grad_norm},
            epoch * len(data_loader) + step,
        )
        writer.add_scalars(
            f"train_step/lr",
            {_prefix: optimizer.param_groups[0]["lr"]},
            epoch * len(data_loader) + step,
        )

        if (step + 1) % cfg.log_freq == 0 or step == len(data_loader) - 1:
            logger.info(
                f"epoch {epoch + 1} / {cfg.epochs} step {step + 1} / {len(data_loader)} loss {loss.item():.4f} loss avg {total_loss / (step + 1):.4f} grad norm {grad_norm:.4f}"
            )

    logger.info(
        f"epoch {epoch + 1} / {cfg.epochs} train loss {total_loss / (step + 1):.4f}"
    )
    return total_loss / (step + 1)


def valid_epoch(model, data_loader, criterion, device, cfg, writer=None, _prefix=""):
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

        outputs = torch.softmax(outputs, dim=1)
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

    return (
        f1,
        total_loss / (step + 1),
        preds,
        labels,
    )


def train_(
    cfg, model, tokenizer, train_df, valid_df, device, writer=None, stage=1, _prefix=""
):
    if len(train_df["label_id"].unique()) != cfg.num_labels:
        logger.warning(
            f"train data labels do not have all labels, {len(train_df['label_id'].unique())} != {cfg.num_labels}"
        )
    if len(valid_df["label_id"].unique()) != cfg.num_labels:
        logger.warning(
            f"valid data labels do not have all labels, {len(valid_df['label_id'].unique())} != {cfg.num_labels}"
        )

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
    criterion = get_criterion(cfg, "focal" if stage == 1 else "focal")
    optimizer = get_optimizer(model, cfg)
    num_train_steps = int(len(train_dt) / cfg.batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    best_f1 = 0
    last_f1 = 0

    for epoch in range(cfg.epochs):
        logger.info(f"epoch {epoch + 1} / {cfg.epochs}")
        start_time = time.time()

        train_epoch(
            epoch,
            model,
            train_dl,
            criterion,
            optimizer,
            scheduler,
            device,
            cfg,
            writer,
            _prefix=_prefix,
        )
        logger.info(
            f"train epoch {epoch + 1} / {cfg.epochs} done in {time.time() - start_time:.2f} seconds"
        )

        start_time = time.time()
        f1_train, loss_train, preds_train, labels_train = valid_epoch(
            model,
            train_dl,
            criterion,
            device,
            cfg,
            writer,
            _prefix=_prefix,
        )
        f1_val, loss_val, preds_val, labels_val = valid_epoch(
            model,
            valid_dl,
            criterion,
            device,
            cfg,
            writer,
            _prefix=_prefix,
        )
        writer.add_scalars(f"valid_epoch/f1", {_prefix: f1_val}, epoch)
        writer.add_scalars(
            f"valid_epoch/acc", {_prefix: (labels_val == preds_val).mean()}, epoch
        )
        writer.add_scalars(f"valid_epoch/loss", {_prefix: loss_val}, epoch)

        writer.add_scalars(f"train_epoch/f1", {_prefix: f1_train}, epoch)
        writer.add_scalars(
            f"train_epoch/acc", {_prefix: (labels_train == preds_train).mean()}, epoch
        )
        writer.add_scalars(f"train_epoch/loss", {_prefix: loss_train}, epoch)

        logger.info(
            f"valid epoch {epoch + 1} / {cfg.epochs} done in {time.time() - start_time:.2f} seconds f1 train {f1_train:.4f} f1 val {f1_val:.4f}"
        )

        if f1_val > best_f1:
            best_f1 = f1_val
            logger.info(f"best fl val score {best_f1:.4f} saving model")
            tokenizer.save_pretrained(cfg.model_path / f"{_prefix}_best")
            torch.save(
                model.state_dict(),
                cfg.model_path / f"{_prefix}_best" / f"model_{stage}.pth",
            )

        if (epoch + 1) % cfg.chpt_freq == 0:
            logger.info(f"saving model at epoch {epoch + 1}")
            tokenizer.save_pretrained(cfg.model_path / f"{_prefix}_chpt_{epoch + 1}")
            torch.save(
                model.state_dict(),
                cfg.model_path / f"{_prefix}_chpt_{epoch + 1}" / f"model_{stage}.pth",
            )

        last_f1 = f1_val

    logger.info(f"best f1 score {best_f1:.4f}")
    tokenizer.save_pretrained(cfg.model_path / f"{_prefix}_final")
    torch.save(
        model.state_dict(), cfg.model_path / f"{_prefix}_final" / f"model_{stage}.pth"
    )

    return best_f1, last_f1


def train_two_stage(cfg_path: str, model_path: str = None):
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

            # TODO: random selection of folds
            data_df.loc[data_df["label_id"] == i, "fold"] = (
                np.arange(len_label) % cfg.num_folds
            )

        data_df["fold"] = data_df["fold"].astype(int)
        # data_df[data_df["label"] == i]["fold"] = np

    else:
        train_df = data_df.sample(frac=cfg.train_frac, random_state=cfg.seed)
        valid_df = data_df.drop(train_df.index).reset_index(drop=True)

    model_path = (
        Path(model_path) if model_path is not None else cfg.model_path / "pretrained"
    )

    if cfg.from_scratch:
        logger.info(f"training from scratch, model name: {cfg.model_name}")
        model_one = get_model(cfg.model_name, 2)
        model_two = get_model(cfg.model_name, cfg.num_labels - 1)
    else:
        raise NotImplementedError

    tokenizer = get_tokenizer(cfg.model_name)

    if cfg.use_tensorboard:
        tb_writer = SummaryWriter(cfg.log_path)
    else:
        tb_writer = None

    if cfg.use_kfold:
        best_f1s_one, last_f1s_one = [], []
        best_f1s_two, last_f1s_two = [], []
        for fold in range(cfg.num_folds):
            logger.info(f"fold {fold + 1} / {cfg.num_folds}")
            train_df = data_df[data_df["fold"] != fold]
            valid_df = data_df[data_df["fold"] == fold]

            if cfg.from_scratch:
                logger.info(f"training from scratch, model name: {cfg.model_name}")
                model_one = get_model(cfg.model_name, 2)
                model_two = get_model(cfg.model_name, cfg.num_labels - 1)
            else:
                raise NotImplementedError

            train_df_one, train_df_two = get_two_stage_data(train_df)
            valid_df_one, valid_df_two = get_two_stage_data(valid_df)

            best_f1_one, last_f1_one = train_(
                cfg,
                model_one,
                tokenizer,
                train_df_one,
                valid_df_one,
                cfg.device,
                tb_writer,
                stage=1,
                _prefix=f"one_{fold}",
            )
            best_f1_two, last_f1_two = train_(
                cfg,
                model_two,
                tokenizer,
                train_df_two,
                valid_df_two,
                cfg.device,
                tb_writer,
                stage=2,
                _prefix=f"two_{fold}",
            )

            best_f1s_one.append(best_f1_one)
            last_f1s_one.append(last_f1_one)
            best_f1s_two.append(best_f1_two)
            last_f1s_two.append(last_f1_two)

            logger.info(f"fold {fold + 1} / {cfg.num_folds} done")

        logger.info(
            f"best f1 one {np.mean(best_f1s_one):.4f} last f1 one {np.mean(last_f1s_one):.4f}"
        )
        logger.info(
            f"best f1 two {np.mean(best_f1s_two):.4f} last f1 two {np.mean(last_f1s_two):.4f}"
        )
    else:
        raise NotImplementedError


def test_two_stage(cfg_path: str, model_path: str):
    cfg = init(cfg_path)
    logger.info(cfg)

    if model_path is None:
        model_path = (
            cfg.model_path / "best" if cfg.use_best else cfg.model_path / "final"
        )

    model_one = load_model(
        model_path, model_name=cfg.model_name, num_labels=2, model_pth="model_1.pth"
    )
    model_two = load_model(
        model_path,
        model_name=cfg.model_name,
        num_labels=cfg.num_labels - 1,
        model_pth="model_2.pth",
    )

    tokenizer = load_tokenizer(model_path)

    test_df = load_data(cfg.data_path, split="test")
    logger.info(test_df.head())

    test_dataset = FSDataset(test_df, tokenizer, is_test=True)
    test_dl = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_one.to(cfg.device)
    model_one.eval()

    preds_one = []
    for step, batch in enumerate(test_dl):
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)

        with torch.no_grad():
            outputs = model_one(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        preds_one.extend(outputs.argmax(dim=1).to("cpu").numpy())

        if (step + 1) % cfg.log_freq == 0 or step == len(test_dl) - 1:
            logger.info(f"step: {step}/{len(test_dl)}")

    preds_one = np.array(preds_one)

    del model_one

    model_two.to(cfg.device)
    model_two.eval()
    preds_two = []
    for step, batch in enumerate(test_dl):
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)

        with torch.no_grad():
            outputs = model_two(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        preds_two.extend(outputs.argmax(dim=1).to("cpu").numpy())

        if (step + 1) % cfg.log_freq == 0 or step == len(test_dl) - 1:
            logger.info(f"step: {step}/{len(test_dl)}")

    preds_two = np.array(preds_two)

    preds = np.zeros_like(preds_one)

    preds[preds_one == 0] = preds_two[preds_one == 0]
    preds[preds_one == 1] = 2

    test_df["label"] = preds

    test_df_export = test_df[["id", "label"]]
    test_df_export.to_csv(
        os.path.join(
            cfg.export_path,
            f"test_submitA_{cfg.model_name}_ep{cfg.epochs}_bs{cfg.batch_size}_{int(time.time())}.csv",
        ),
        index=False,
    )
