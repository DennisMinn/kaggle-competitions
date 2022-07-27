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
    def on_fit_start(self, trainer, pl_module):
        wandb.define_metric("step")
        wandb.define_metric("epoch")
        self.train_step, self.val_step = 1, 1
        self.train_epoch, self.val_epoch = 1, 1

        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()

    def on_train_batch_end(self,
                           trainer,
                           pl_module,
                           outputs,
                           batch,
                           batch_idx,
                           dataloader_idx):
        self.train_loss.update(outputs["loss"], 1)
        if self.train_step % 100 == 0:
            self.log_dict({
                "train/loss": self.train_loss.avg_value,
                "step": self.train_step,
            })

        self.train_step += 1

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx):
        self.val_loss.update(outputs["loss"], 1)
        if self.val_step % 100 == 0:
            self.log_dict({
                "validation/loss": self.val_loss.avg_value,
                "loss": self.val_loss.avg_value,
                "step": self.val_step,
            })

        self.val_step += 1

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_dict({
            "train/loss": self.train_loss.avg_value,
            "epoch": self.train_epoch
        })
        self.train_loss.reset()
        self.train_epoch += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_dict({
            "validation/loss": self.val_loss.avg_value,
            "loss": self.val_loss.avg_value,
            "epoch": self.val_epoch
        })
        self.val_loss.reset()
        self.val_epoch += 1

    def on_fit_end(self, trainer, pl_module):
        wandb.finish()
