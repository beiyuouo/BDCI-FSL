import os
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.init import init
from utils.data import load_data
from core.model import load_model, load_tokenizer
from core.dataset import FSDataset


def testA(cfg_path: str, model_path: str):
    cfg = init(cfg_path)
    logger.info(cfg)

    if model_path is None:
        model_path = (
            cfg.model_path / "best" if cfg.use_best else cfg.model_path / "final"
        )

    model = load_model(model_path)
    logger.info(model)

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

    model.to(cfg.device)
    model.eval()

    preds = []
    for step, batch in enumerate(test_dl):
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        token_type_ids = batch["token_type_ids"].to(cfg.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        preds.extend(outputs.logits.argmax(dim=1).to("cpu").numpy())

        if step % cfg.log_freq == 0:
            logger.info(f"step: {step}/{len(test_dl)}")

    preds = np.array(preds)

    test_df["label"] = preds

    test_df_export = test_df[["id", "label"]]
    test_df_export.to_csv(
        os.path.join(
            cfg.export_path, f"test_submit_A_ep{cfg.epochs}_bs{cfg.batch_size}.csv"
        ),
        index=False,
    )
