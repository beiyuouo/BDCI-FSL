from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding,
)
import torch
from torch.optim import AdamW


def get_model(model_name: str = "hfl/chinese-roberta-wwm-ext", num_labels: int = 36):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def get_tokenizer(model_name: str = "hfl/chinese-roberta-wwm-ext"):
    return AutoTokenizer.from_pretrained(model_name)


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters


def get_optimizer(model, cfg):
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=cfg.encoder_lr,
        decoder_lr=cfg.decoder_lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer = AdamW(
        optimizer_parameters, lr=cfg.encoder_lr, eps=cfg.eps, betas=cfg.betas
    )
    return optimizer


def get_scheduler(cfg, optimizer, num_train_steps):
    cfg.num_warmup_steps = cfg.num_warmup_steps * num_train_steps
    if cfg.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    elif cfg.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles,
        )
    return scheduler


def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer
