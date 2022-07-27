import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from utils import util
from datamodule.effectiveness_datamodule import EffectivenessDataModule
from model.effectiveness_model import EffectivenessModel
from logger.logger import EffectivenessLogger

config = {
    "input_dir": "data/train.csv",
    "output_dir": "saved",
    "num_workers": 8,
    "batch_size": 40,
    "max_len": 256,
    "model_name": "distilbert-base-cased",
    "output_dim": 3,
    "num_epochs": 4,
    "learning_rate": 2e-5,
    "betas": (0.9, 0.999),
    "eps": 1e-6,
    "num_warmup_steps": 0,
}

pl.utilities.seed.seed_everything(seed=42)
datamodule = EffectivenessDataModule(config)
datamodule.setup("fit")

model = EffectivenessModel(**config)
logger = EffectivenessLogger("Feedback Prize", util.timestamp(), "saved/logs")
lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(gpus=1,
                     max_epochs=config["num_epochs"],
                     callbacks=[logger, lr_monitor],
                     num_sanity_val_steps=0,
                     default_root_dir="saved/logs")

trainer.fit(model, datamodule)
