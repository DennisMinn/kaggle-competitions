import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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

        self.discourse_text = df["discourse_text"]
        self.discourse_type = df["discourse_type"]

        if self.train:
            self.label = df["discourse_effectiveness"]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        discourse_text = self.discourse_text[index]
        discourse_type = self.discourse_type[index]

        text = discourse_type + self.tokenizer.sep_token + discourse_text
        batch_encoding = self.tokenizer.encode_plus(
            text=text,
            padding="max_length",
            max_length=self.max_len,
            truncation=True
        )

        inputs = {}
        inputs["batch_encoding"] = {k: torch.tensor(v, dtype=torch.long)
                                    for k, v in batch_encoding.items()}
        if self.train:
            label = label2id[self.label[index]]
            inputs["label"] = torch.tensor([label], dtype=torch.long)

        return inputs


class EffectivenessDataModule(pl.LightningDataModule):
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(self.config["input_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

    def setup(self, stage=None):
        if stage in (None, "fit"):
            dataset = EffectivenessDataset(self.df, self.tokenizer,
                                           max_len=self.config["max_len"],
                                           train=True)

            train_len = int(len(dataset) * 0.8)
            val_len = len(dataset) - train_len

            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [train_len, val_len]
            )

        if stage in (None, "test"):
            dataset = EffectivenessDataset(self.df, self.tokenizer,
                                           max_len=self.config["max_len"],
                                           train=False)

            self.test_dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            pin_memory=True)
