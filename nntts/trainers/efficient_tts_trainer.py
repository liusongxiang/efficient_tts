import logging
import os
import sys

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from nntts.utils.plotting import plots
matplotlib.use("Agg")


class EfficientTTSTrainer(object):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(self,
                 steps,
                 epochs,
                 data_loader,
                 sampler,
                 model,
                 optimizer,
                 scheduler,
                 config,
                 device=torch.device("cpu"),
                 ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         desc="[train]")
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(), 
            # "scheduler": self.scheduler.state_dict(), 
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        if self.config["distributed"]:
            state_dict["model"] = self.model.module.state_dict() 
        else:
            state_dict["model"] = self.model.state_dict() 
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state_dict["scheduler"])

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        datas = [x.to(self.device) for x in batch]
        if len(datas) >= 6:
            text, text_lengths, duration, mel, mel_lengths, spkids = [x.to(self.device) for x in batch]

            # torch.autograd.set_detect_anomaly(True)
            loss, stats = self.model(
                text=text,
                text_lengths=text_lengths,
                speech=mel,
                speech_lengths=mel_lengths,
                durations=duration,
                spembs=spkids,
            )
        else:
            text, text_lengths, mel, mel_lengths = datas
            loss, stats, *_ = self.model(
                text=text,
                text_lengths=text_lengths,
                speech=mel,
                speech_lengths=mel_lengths,
            )
            
        # update stats
        self.total_train_loss["train/loss"] += stats["loss"]
        self.total_train_loss["train/mel_loss"] += stats["mel_loss"]
        self.total_train_loss["train/dur_loss"] += stats["duration_loss"]
        
        # update model
        self.optimizer.zero_grad()
        loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["grad_norm"])
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch, plot=False):
        """Evaluate model one step."""
        datas = [x.to(self.device) for x in batch]
        if len(datas) >= 6:
            text, text_lengths, duration, mel, mel_lengths, spkids = [x.to(self.device) for x in batch]

            # torch.autograd.set_detect_anomaly(True)
            loss, stats = self.model(
                text=text,
                text_lengths=text_lengths,
                speech=mel,
                speech_lengths=mel_lengths,
                durations=duration,
                spembs=spkids,
            )
        else:
            text, text_lengths, mel, mel_lengths = datas
            loss, stats, imv, alpha, mel_pred, mel_gt = self.model(
                text=text,
                text_lengths=text_lengths,
                speech=mel,
                speech_lengths=mel_lengths,
            )
            if plot:
                plots(imv, alpha, mel_pred, mel_gt, self.steps, self.config["outdir"])
        # update stats
        self.total_eval_loss["eval/loss"] += stats["loss"]
        self.total_eval_loss["eval/mel_loss"] += stats["mel_loss"]
        self.total_eval_loss["eval/dur_loss"] += stats["duration_loss"]
    
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        self.model.eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["dev"], desc="[eval]"), 1):
            # eval one step
            if eval_steps_per_epoch == 1:
                plot = True
            self._eval_step(batch, plot)

        logging.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        self.model.train()
    
    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True
