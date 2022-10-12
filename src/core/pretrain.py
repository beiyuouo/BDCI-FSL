import pandas as pd
from loguru import logger
import torch
import time
from torch.utils.data import DataLoader

from utils.init import init
from utils.data import load_data
from core.model import (
    load_maskedlm_model,
    get_maskedlm_model,
    get_tokenizer,
    get_optimizer,
    get_scheduler,
)
from core.dataset import FSDataset


def train_epoch(epoch, model, data_loader, optimizer, scheduler, device, cfg):
    model.train()
    training_loss = 0.0
    for step, batch in enumerate(data_loader):
        input_ids = batch["inputs"].squeeze(0).to(device)
        labels = batch["labels"].squeeze(0).to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        training_loss += loss.item()

        if (step + 1) % cfg.log_freq == 0 or step == len(data_loader) - 1:
            logger.info(
                f"epoch {epoch + 1} / {cfg.epochs}, step {step + 1} / {len(data_loader)}, loss {loss.item():.4f}"
            )


def pretrain(cfg_path: str = "config/hyps.yaml", model_path: str = None):
    cfg = init(cfg_path)
    logger.info(f"config: {cfg}")

    if model_path is None:
        model = get_maskedlm_model(cfg.model_name)
    else:
        model = load_maskedlm_model(model_path)

    tokenizer = get_tokenizer(cfg.model_name)

    logger.info(model)

    data_dfs = load_data(cfg.data_path, split="all")
    data_df = pd.concat(data_dfs, axis=0, ignore_index=True)

    logger.info(f"train data size: {len(data_df)}")
    logger.info(data_df.head())

    train_data = data_df

    train_ds = FSDataset(train_data, tokenizer, is_test=True, mlm=True, cfg=cfg)

    train_dl = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    model.to(cfg.device)

    optimizer = get_optimizer(model, cfg)
    num_train_steps = int(len(train_ds) / cfg.batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    for epoch in range(cfg.pretrain.epochs):
        logger.info(f"epoch {epoch + 1} / {cfg.epochs}")
        start_time = time.time()

        train_epoch(epoch, model, train_dl, optimizer, scheduler, cfg.device, cfg)
        logger.info(
            f"train epoch {epoch + 1} / {cfg.epochs} done in {time.time() - start_time:.2f} seconds"
        )

    model.save_pretrained(cfg.model_path / "pretrained")


if __name__ == "__main__":
    pretrain()
