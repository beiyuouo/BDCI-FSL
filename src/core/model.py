from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForMaskedLM,
    BertModel,
)
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding,
)
import torch
import torch.nn as nn
from torch.optim import AdamW


class BDCIModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        last_hidden_state, pooler_output = (
            outputs.last_hidden_state,
            outputs.pooler_output,
        )

        pooled_output = self.dropout(pooler_output)
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)

        return logits

    def get_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        last_hidden_state, pooler_output = (
            outputs.last_hidden_state,
            outputs.pooler_output,
        )

        return last_hidden_state, pooler_output

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False


def get_model(model_name: str = "hfl/chinese-bert-wwm-ext", num_labels: int = 36):
    return BDCIModel(model_name, num_labels)


def get_tokenizer(model_name: str = "hfl/chinese-bert-wwm-ext"):
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


def load_model(model_path, model_name="hfl/chinese-bert-wwm-ext", num_labels: int = 36):
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_path, num_labels=num_labels
    # )
    model = BDCIModel(model_path, num_labels)
    model.load_state_dict(torch.load(model_path / "model.pth"), strict=False)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def get_maskedlm_model(model_name: str = "hfl/chinese-roberta-wwm-ext"):
    return AutoModelForMaskedLM.from_pretrained(model_name)


def load_maskedlm_model(model_path):
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    return model
