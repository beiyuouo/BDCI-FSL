import os
import time
from pathlib import Path
from loguru import logger
import torch
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
        token_type_ids = batch["token_type_ids"].to(device)
        input_ids = batch["input_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
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

        if step % cfg.log_freq == 0 or step == len(data_loader) - 1:
            logger.info(
                f"epoch {epoch + 1} / {cfg.epochs} step {step + 1} / {len(data_loader)} loss {loss.item():.4f} loss avg {total_loss / (step + 1):.4f} grad norm {grad_norm:.4f}"
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
                token_type_ids=token_type_ids,
                labels=label,
            )

        loss = outputs.loss
        total_loss += loss.item()

        preds.extend(outputs.logits.argmax(dim=1).to("cpu").numpy())
        labels.extend(label.to("cpu").numpy())

        if step % cfg.log_freq == 0 or step == len(data_loader) - 1:
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
        model = get_model(cfg.model_name, cfg.num_labels)
    else:
        model = load_model(model_path)

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
        f1 = valid_epoch(model, valid_dl, cfg.device, cfg)
        logger.info(
            f"valid epoch {epoch + 1} / {cfg.epochs} done in {time.time() - start_time:.2f} seconds f1 score {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            logger.info(f"best f1 score {best_f1:.4f} saving model")
            model.save_pretrained(cfg.model_path / "best")
            tokenizer.save_pretrained(cfg.model_path / "best")

        if (epoch + 1) % cfg.chpt_freq == 0:
            logger.info(f"saving model at epoch {epoch + 1}")
            model.save_pretrained(cfg.model_path / f"chpt_{epoch + 1}")
            tokenizer.save_pretrained(cfg.model_path / f"chpt_{epoch + 1}")

    logger.info(f"best f1 score {best_f1:.4f}")
    model.save_pretrained(cfg.model_path / "final")
    tokenizer.save_pretrained(cfg.model_path / "final")
