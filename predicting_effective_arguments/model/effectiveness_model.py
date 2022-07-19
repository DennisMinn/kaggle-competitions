import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModel
from model.metrics import accuracy, precision, recall, f1


class EffectivenessModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model_config = AutoConfig.from_pretrained(
            self.hparams["model_name"])
        self.encoder = AutoModel.from_config(self.model_config)
        self.fc = nn.Linear(self.model_config.hidden_size,
                            self.hparams["output_dim"])

    def forward(self, inputs):
        out = self.encoder(**inputs)
        out = out.last_hidden_state[:, 0, :]
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        output = self.forward(batch["batch_encoding"])
        label = batch["label"]

        loss = F.cross_entropy(output, label)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch["batch_encoding"])
        label = batch["label"]

        loss = F.cross_entropy(output, label)
        pred = torch.argmax(output.detach(), dim=-1)

        metrics = {
            "accuracy": accuracy(label, pred),
            "precision": precision(label, pred),
            "recall": recall(label, pred),
            "f1": f1(label, pred)
        }

        return {"loss": loss, "pred": pred, "metrics": metrics}

    def test_step(self, batch, batch_idx):
        output = self.forward(batch["batch_encoding"])
        return output

    def configure_optimizers(self):
        """
        initializes optimizers
        """
        optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams['learning_rate'],
                eps=self.hparams['eps'],
                betas=self.hparams['betas'])

        scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams['num_warmup_steps'],
                num_training_steps=self.hparams['num_training_steps'])

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
