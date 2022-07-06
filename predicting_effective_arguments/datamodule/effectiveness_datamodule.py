import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl

label2id = {
    "Ineffective": 0,
    "Adequate": 1,
    "Effective": 2
}

id2label = {
    0: "Ineffective",
    1: "Adequate",
    2: "Effective"
}


class EffectivenessDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, train=True):
        self.len = df.shape[0]
        self.train = train

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.discourse_text = df['discourse_text']
        self.discourse_type = df['discourse_type']

        if self.train:
            self.label = df['discourse_effectiveness']

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        discourse_text = self.discourse_text[index]
        discourse_type = self.discourse_type[index]

        label = label2id[self.label[index]]

        text = discourse_type + self.tokenizer.sep_token + discourse_text
        batch_encoding = self.tokenizer.encode_plus(
            text=text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True
        )

        inputs = {}
        inputs['batch_encoding'] = {k: torch.tensor(v, dtype=torch.long)
                                    for k, v in batch_encoding.items()}
        if self.train:
            inputs['label'] = torch.tensor([label], dtype=torch.long)

        return inputs
