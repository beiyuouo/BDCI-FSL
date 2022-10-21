from csv import writer
from loguru import logger
import numpy as np
import os
import pandas as pd
import time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans

from core.dataset import FSDataset
from core.model import get_model, load_model, get_tokenizer
from utils.data import load_data
from utils.init import init


def cluster(cfg_path: str, model_path: str = None):
    cfg = init(cfg_path)
    logger.info(cfg)

    data_df = load_data(cfg.data_path, split="train")
    logger.info(data_df.head())

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
        writer = SummaryWriter(cfg.log_path)
    else:
        writer = None

    data_ds = FSDataset(data_df, tokenizer, cfg=cfg)
    data_dl = DataLoader(
        data_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    model.eval()
    model.to(cfg.device)

    embs = []
    for i, batch in enumerate(data_dl):
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)

        with torch.no_grad():
            hidden_state, pooled_emb = model.get_embeddings(
                input_ids=input_ids, attention_mask=attention_mask
            )

        embs.extend(pooled_emb.to("cpu").numpy())

        if (i + 1) % cfg.log_freq == 0 or i == len(data_dl) - 1:
            logger.info(f"step: {i+1}/{len(data_dl)}")

    embs = np.array(embs)

    np.save(Path(cfg.data_path) / "train_embeddings.npy", embs)

    label_avg_embeds = []

    for i, group in data_df.groupby("label_id"):
        label_avg_embeds.append(np.mean(embs[group.index], axis=0))
        print(i, len(group))

    label_avg_embeds = np.array(label_avg_embeds)
    np.save(os.path.join(cfg.data_path, "label_embeddings.npy"), label_avg_embeds)

    test_df = load_data(cfg.data_path, split="test")
    test_ds = FSDataset(test_df, tokenizer, is_test=True, cfg=cfg)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    test_embs = []

    for i, batch in enumerate(test_dl):
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)

        with torch.no_grad():
            hidden_state, pooled_emb = model.get_embeddings(
                input_ids=input_ids, attention_mask=attention_mask
            )

        test_embs.extend(pooled_emb.to("cpu").numpy())

        if (i + 1) % cfg.log_freq == 0 or i == len(test_dl) - 1:
            logger.info(f"step: {i+1}/{len(test_dl)}")

    test_embs = np.array(test_embs)

    np.save(Path(cfg.data_path) / "test_embeddings.npy", test_embs)

    clusterer = KMeans(n_clusters=cfg.num_labels, random_state=cfg.seed)
    clusterer.fit(test_embs)

    # cluster centers cal

    test_df["label"] = clusterer.labels_
    test_df.to_csv(
        Path(cfg.data_path) / f"test_submit_A_cluster_kmeans_{int(time.time())}.csv",
        index=False,
    )

    # similarity matrix calculation using cosine similarity
    sim_matrix = np.array(
        [
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            for i, a in enumerate(test_embs)
            for j, b in enumerate(label_avg_embeds)
        ]
    ).reshape(len(test_embs), cfg.num_labels)

    # get 3 most similar labels
    preds = (-sim_matrix).argsort()[:, :1]

    test_df["label"] = preds

    logger.info(f"preds shape: {preds.shape}")

    test_df_export = test_df[["id", "label"]]
    test_df_export.to_csv(
        os.path.join(
            cfg.export_path,
            f"test_submit_A_cluster_{int(time.time())}.csv",
        ),
        index=False,
    )
