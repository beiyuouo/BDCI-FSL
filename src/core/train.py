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


def train_epoch(epoch, model, data_loader, optimizer, scheduler, device, cfg):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(data_loader):
        label = batch["label"].to(device)
        mask = batch["attention_mask"].to(device)
        input_ids = batch["input_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=mask,
            labels=label,
        )

        loss = outputs.loss

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


def valid_epoch(model, data_loader, device, cfg):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []

    for step, batch in enumerate(data_loader):
        label = batch["label"].to(device)
        mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        input_ids = batch["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=mask,
                labels=label,
            )

        loss = outputs.loss
        total_loss += loss.item()

        preds.extend(outputs.logits.argmax(dim=1).to("cpu").numpy())
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


def train(cfg_path: str, model_path: str = None):
    cfg = init(cfg_path)
    logger.info(cfg)

    data_df = load_data(cfg.data_path, split="train")
    logger.info(data_df.head())

    train_data = data_df.sample(frac=0.9, random_state=cfg.seed)
    valid_data = data_df.drop(train_data.index).reset_index(drop=True)

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


def train_full(cfg_path: str, model_path: str = None):
    cfg = init(cfg_path)
    logger.info(cfg)

    data_df = load_data(cfg.data_path, split="train")
    logger.info(data_df.head())

    model_path = (
        Path(model_path) if model_path is not None else cfg.model_path / "pretrained"
    )

    if cfg.from_scratch:
        model = get_model(cfg.model_name, cfg.num_labels)
    else:
        model = load_model(model_path, cfg.num_labels)

    tokenizer = get_tokenizer(cfg.model_name)

    train_dataset = FSDataset(data_df, tokenizer)

    train_dl = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
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

        f1_train = valid_epoch(model, train_dl, cfg.device, cfg)

        if (epoch + 1) % cfg.chpt_freq == 0:
            logger.info(f"saving model at epoch {epoch + 1}")
            model.save_pretrained(cfg.model_path / f"chpt_{epoch + 1}")
            tokenizer.save_pretrained(cfg.model_path / f"chpt_{epoch + 1}")

        if f1_train > best_f1:
            best_f1 = f1_train
            logger.info(f"best f1 score {best_f1:.4f} saving model")
            model.save_pretrained(cfg.model_path / "best")
            tokenizer.save_pretrained(cfg.model_path / "best")

    model.save_pretrained(cfg.model_path / "final")
    tokenizer.save_pretrained(cfg.model_path / "final")


def train_epoch_us(epoch, model, unlabeled_dl, optimizer, scheduler, device, cfg):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(unlabeled_dl):
        mask = batch["attention_mask"].to(device)
        input_ids = batch["input_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=mask,
        )

        fake_labels = outputs.logits.argmax(dim=1).detach()

        loss = F.cross_entropy(outputs.logits, fake_labels)

        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm
        )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if (step + 1) % cfg.log_freq == 0 or step == len(unlabeled_dl) - 1:
            logger.info(
                f"epoch {epoch + 1} / {cfg.epochs} step {step + 1} / {len(unlabeled_dl)} loss {loss.item():.4f} loss avg {total_loss / (step + 1):.4f} grad norm {grad_norm:.4f}"
            )

    logger.info(
        f"epoch {epoch + 1} / {cfg.epochs} unsupervised loss {total_loss / (step + 1):.4f}"
    )


def train_epoch_ssl(
    epoch, model, labeled_dl, unlabeled_dl, optimizer, scheduler, device, cfg
):
    train_epoch(epoch, model, labeled_dl, optimizer, scheduler, device, cfg)
    train_epoch_us(epoch, model, unlabeled_dl, optimizer, scheduler, device, cfg)


def train_ssl_base(cfg_path: str, model_path: str = None):
    cfg = init(cfg_path)
    logger.info(cfg)

    labeled_df = load_data(cfg.data_path, split="train")
    unlabeled_df = load_data(cfg.data_path, split="test")

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

    labeled_ds = FSDataset(labeled_df, tokenizer)
    unlabeled_ds = FSDataset(unlabeled_df, tokenizer, is_test=True)

    labeled_dl = DataLoader(
        labeled_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    unlabeled_dl = DataLoader(
        unlabeled_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    logger.info(f"labeled dataset size {len(labeled_ds)}")
    logger.info(f"unlabeled dataset size {len(unlabeled_ds)}")

    model.to(cfg.device)

    optimizer = get_optimizer(model, cfg)
    num_train_steps = int(len(labeled_ds) / cfg.batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    best_f1 = 0

    for epoch in range(cfg.epochs):
        logger.info(f"epoch {epoch + 1} / {cfg.epochs}")
        start_time = time.time()

        train_epoch_ssl(
            epoch,
            model,
            labeled_dl,
            unlabeled_dl,
            optimizer,
            scheduler,
            cfg.device,
            cfg,
        )
        logger.info(
            f"train epoch {epoch + 1} / {cfg.epochs} done in {time.time() - start_time:.2f} seconds"
        )

        f1_train = valid_epoch(model, labeled_dl, cfg.device, cfg)

        if (epoch + 1) % cfg.chpt_freq == 0:
            logger.info(f"saving model at epoch {epoch + 1}")
            model.save_pretrained(cfg.model_path / f"chpt_{epoch + 1}")
            tokenizer.save_pretrained(cfg.model_path / f"chpt_{epoch + 1}")

    for epoch in range(cfg.epochs):
        logger.info(f"epoch {epoch + 1} / {cfg.epochs}")
        start_time = time.time()

        train_epoch()

    model.save_pretrained(cfg.model_path / "final")
    tokenizer.save_pretrained(cfg.model_path / "final")


def train_ssl(cfg_path: str, model_path: str = None):
    cfg = init(cfg_path)
    logger.info(cfg)

    labeled_df = load_data(cfg.data_path, split="train")
    unlabeled_df = load_data(cfg.data_path, split="test")

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

    labeled_dt = labeled_df.sample(frac=0.9, random_state=cfg.seed)
    valid_dt = labeled_df.drop(labeled_dt.index).reset_index(drop=True)

    labeled_ds = FSDataset(labeled_dt, tokenizer)
    valid_ds = FSDataset(valid_dt, tokenizer)
    unlabeled_ds = FSDataset(unlabeled_df, tokenizer, is_test=True)

    labeled_dl = DataLoader(
        labeled_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    unlabeled_dl = DataLoader(
        unlabeled_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    logger.info(f"labeled dataset size {len(labeled_ds)}")
    logger.info(f"unlabeled dataset size {len(unlabeled_ds)}")

    model.to(cfg.device)

    optimizer = get_optimizer(model, cfg)
    num_train_steps = int(len(labeled_ds) / cfg.batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    label_iter = iter(labeled_dl)
    unlabel_iter = iter(unlabeled_dl)

    total_steps = int(len(labeled_ds) / cfg.batch_size * cfg.epochs)
    epoch_step = int(len(labeled_ds) / cfg.batch_size)
    # total_steps = epoch_step
    epoch = 0

    best_f1 = 0
    start_time = time.time()

    for step in range(total_steps):
        # logger.info(f"step {step + 1} / {total_steps}")

        train_loss = 0.0
        model.train()

        try:
            labeled_batch = next(label_iter)
        except StopIteration:
            label_iter = iter(labeled_dl)
            labeled_batch = next(label_iter)

        try:
            unlabeled_batch = next(unlabel_iter)
        except StopIteration:
            unlabel_iter = iter(unlabeled_dl)
            unlabeled_batch = next(unlabel_iter)

        labeled_ids = labeled_batch["input_ids"].to(cfg.device)
        labeled_mask = labeled_batch["attention_mask"].to(cfg.device)
        labeled_labels = labeled_batch["label"].to(cfg.device)

        unlabeled_ids = unlabeled_batch["input_ids"].to(cfg.device)
        unlabeled_mask = unlabeled_batch["attention_mask"].to(cfg.device)

        optimizer.zero_grad()
        labeled_ouputs = model(
            labeled_ids, attention_mask=labeled_mask, labels=labeled_labels
        )
        unlabeled_ouputs = model(unlabeled_ids, attention_mask=unlabeled_mask)

        fake_labels = torch.argmax(unlabeled_ouputs.logits, dim=1).detach()

        # sharpening
        # fake_labels = torch.softmax(unlabeled_ouputs.logits / cfg.ssl.T, dim=1).argmax(
        #     dim=1
        # )

        fake_labels = fake_labels.float()
        # fake_labels to one-hot
        fake_labels = F.one_hot(fake_labels.to(torch.int64), cfg.num_labels).float()

        # logger.info(f"labeled loss {labeled_ouputs.loss.item():.4f}")
        # logger.info(f"unlabeled fake labels {fake_labels}")
        # logger.info(
        #     f"unlabeled loss {F.mse_loss(unlabeled_ouputs.logits.float(), fake_labels)}"
        # )

        loss = labeled_ouputs.loss + cfg.ssl.lambda_u * F.mse_loss(
            unlabeled_ouputs.logits.float(), fake_labels
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

        if (step + 1) % cfg.log_freq == 0:
            logger.info(
                f"train step {step + 1} / {total_steps} done in {time.time() - start_time:.2f} seconds with loss {train_loss / ((step + 1) % epoch_step + 1):.4f}"
            )

        if (step + 1) % epoch_step == 0:
            epoch += 1
            logger.info(f"epoch {epoch + 1} / {cfg.epochs}")
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0

            model.eval()

            f1_train = valid_epoch(model, labeled_dl, cfg.device, cfg)
            f1_valid = valid_epoch(model, valid_dl, cfg.device, cfg)
            logger.info(
                f"train f1 {f1_train:.4f} valid f1 {f1_valid:.4f} in {time.time() - start_time:.2f} seconds"
            )

            if f1_valid > best_f1:
                best_f1 = f1_valid
                logger.info(f"save model with f1 {best_f1:.4f}")
                model.save_pretrained(cfg.model_path / "best_ssl")
                tokenizer.save_pretrained(cfg.model_path / "best_ssl")

            if (epoch + 1) % cfg.chpt_freq == 0:
                logger.info(f"save model at epoch {epoch + 1}")
                model.save_pretrained(cfg.model_path / f"ssl_{epoch + 1}")
                tokenizer.save_pretrained(cfg.model_path / f"ssl_{epoch + 1}")

    model.save_pretrained(cfg.model_path / "final_ssl")
    tokenizer.save_pretrained(cfg.model_path / "final_ssl")

    return cfg.model_path / "final_ssl"
