import torch
from torch.utils.data import Dataset


class FSDataset(Dataset):
    def __init__(self, df, tokenizer, is_test=False, mlm=False, cfg=None):
        self.title = df["title"].values
        self.assignee = df["assignee"].values
        self.abstract = df["abstract"].values

        if not is_test:
            self.label = df["label_id"].values

        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.is_test = is_test
        self.mlm = mlm
        self.cfg = cfg

    def __len__(self):
        return len(self.title)

    def __get_input_text__(self, item):
        title = self.title[item]
        assignee = self.assignee[item]
        abstract = self.abstract[item]
        input_text = title + self.sep_token + assignee + self.sep_token + abstract 

        return input_text

    def __get_single_item__(self, item):
        input_text = self.__get_input_text__(item)

        inputs = self.tokenizer(
            input_text, truncation=True, max_length=512, padding="max_length"
        )
        if not self.is_test:
            label = int(self.label[item])
        else:
            label = 0

        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

    def __get_mlm_item__(self, item):
        input_text = self.__get_input_text__(item)

        # tokenize batch data
        feats = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # get masked input ids and labels
        inputs, labels = self.mask_tokens(feats["input_ids"])

        return {
            "inputs": inputs,
            "labels": labels,
        }

    def __getitem__(self, item):
        if self.mlm:
            return self.__get_mlm_item__(item)

        return self.__get_single_item__(item)

    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        Code reference: https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247512675&idx=2&sn=0c0a9908681d1b8435d70e475e388144&chksm=eb53a6f0dc242fe65ce388ceae845699bf0deac78c234a3dd54a714ccdff1f4c3c30eacaeb96&mpshare=1&scene=23&srcid=1011f9kK9Ku85TMmdO65ccXj&sharer_sharetime=1665485819713&sharer_shareid=b0947a15cb5f5538a01712000e15ef32#rd
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.cfg.mlm_probability)
        if self.cfg.special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = self.cfg.special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.cfg.prob_replace_mask)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.cfg.prob_replace_rand / (1 - self.cfg.prob_replace_mask)
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, current_prob)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
