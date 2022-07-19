from pytorch_lightning.callbacks import Callback
import wandb


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_value = 0
        self.avg_value = 0
        self.count = 0

    def update(self, value, n):
        self.total_value += value * n
        self.count += n
        self.avg_value = self.total_value / self.count


class EffectivenessLogger(Callback):
    def __init__(self, project, name, fpath):
        super().__init__()
        self.project = project
        self.name = name
        self.fpath = fpath

        self.train_step = 0
        self.val_step = 0

        self.train_metrics = {
            "train/loss": AverageMeter()
        }

        self.val_metrics = {
            "validation/loss": AverageMeter(),
            "validation/accuracy": AverageMeter(),
            "validation/precision": AverageMeter(),
            "validation/recall": AverageMeter(),
            "validation/f1": AverageMeter()
        }

        for metric in self.train_metrics.keys():
            wandb.define_metric(metric, step_metric="train step")

        for metric in self.val_metrics.keys():
            wandb.define_metric(metric, step_metric="validation step")

        self.predictions = []

    def on_fit_start(self, trainer, pl_module):
        wandb.init(project=self.project,
                   name=self.name,
                   config=pl_module.hparams,
                   dir=self.fpath)

    def on_train_batch_end(self,
                           trainer,
                           pl_module,
                           outputs,
                           batch,
                           batch_idx,
                           dataloader_idx):
        n = batch["label"].shape[0]
        self.train_metrics["loss"].update(outputs["loss"], n)

        if self.train_step != 0 and self.train_step % 100 == 0:
            for metric, avg_meter in self.train_metrics.items():
                wandb.log({metric: avg_meter.avg_value,
                           "train step": self.train_step})

        self.train_step += 1

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx):
        n = batch["label"].shape[0]
        self.val_metrics["loss"].update(outputs["loss"], n)

        for metric, value in outputs["metrics"].items():
            self.val_metrics[metric].update(value, n)

        if self.val_step != 0 and self.val_step % 100 == 0:
            for metric, avg_meter in self.val_metrics.items():
                wandb.log({metric: avg_meter.avg_value,
                           "validation step": self.val_step})

        if batch_idx == 0:
            self.predictions = []
        else:
            self.predictions.append(outputs["pred"].cpu())

        self.val_step += 1

    def on_train_epoch_end(self, trainer, pl_module):
        for metric, avg_meter in self.train_metrics.items():
            wandb.log({f"train/{metric}": avg_meter.avg_value,
                       "trian step": self.train_step})
            avg_meter.reset()

    def on_validation_epoch_end(self, trainer, pl_module):
        for metric, avg_meter in self.val_metrics.items():
            wandb.log({f"validation/{metric}": avg_meter.avg_value,
                       "validation step": self.val_step})
            avg_meter.reset()

    def on_fit_end(self, trainer, pl_module):
        wandb.finish()
