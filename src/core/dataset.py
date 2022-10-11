import torch
from torch.utils.data import Dataset


class FSDataset(Dataset):
    def __init__(self, df, tokenizer, is_test=False):
        self.title = df["title"].values
        self.assignee = df["assignee"].values
        self.abstract = df["abstract"].values

        if not is_test:
            self.label = df["label_id"].values

        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.is_test = is_test

    def __len__(self):
        return len(self.title)

    def __getitem__(self, item):
        if not self.is_test:
            label = int(self.label[item])
        else:
            label = 0

        title = self.title[item]
        assignee = self.assignee[item]
        abstract = self.abstract[item]
        input_text = title + self.sep_token + assignee + self.sep_token + abstract
        inputs = self.tokenizer(
            input_text, truncation=True, max_length=400, padding="max_length"
        )
        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
