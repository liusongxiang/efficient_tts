#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Songxiang Liu
#  MIT License (https://opensource.org/licenses/MIT)

"""Train entry for all experiments."""

import argparse
import logging
import os
import sys

import numpy as np
import soundfile as sf
import torch
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

import nntts
import nntts.models
import nntts.optimizers
import nntts.schedulers
import nntts.trainers
import nntts.datasets


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train TTS model (See detail in nntts/bin/train.py).")
    
    parser.add_argument("--train_fid_scp", default=None, type=str,
                        help="fid scp file for training.")
    parser.add_argument("--dev_fid_scp", default=None, type=str,
                        help="fid scp file for training.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--pretrain", default="", type=str, nargs="?",
                        help="checkpoint file path to load pretrained params. (default=\"\")")
    parser.add_argument("--resume", default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explictly specify.")
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = nntts.__version__  # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    # NOTE: ugly current hack!!!
    dataset_class = getattr(
        nntts.datasets,
        config.get("dataset_type", "TextMelDurationSpkidDataset")
    )
    data_related_params = config.get("dataset_params", {})
    train_dataset = dataset_class(
        meta_file=args.train_fid_scp,
        **data_related_params,
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")

    dev_dataset = dataset_class(
        meta_file=args.dev_fid_scp,
        **data_related_params,
    )
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collate_function_class = getattr(
        nntts.datasets,
        config.get("collate_fn_type", "TTSCollate"),
    )
    collate_function_related_params = config.get("collate_fn_params", {})
    collate_function = collate_function_class(
        **collate_function_related_params,
    )
    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler
        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,  # NOTE(sx)
            collate_fn=collate_function,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False if args.distributed else True,
            collate_fn=collate_function,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models and optimizers
    model_class = getattr(
        nntts.models,
        config["model_name"],
    )
    if not "DurationModel" in config["model_name"]:
        try:
            model = model_class(idim=len(dataset["train"].phn_map),
                                **config["model_params"]).to(device) 
        except:
            model = model_class(**config["model_params"]).to(device) 
            
    else:
        model = model_class(**config["model_params"]).to(device)
    # Define losses
    # loss_criterion = FastSpeechLoss()
   
    ## =================================================== ##
    optimizer_class = getattr(
        nntts.optimizers,
        config.get("optimizer_type", "Adam"),
    )
    optimizer = optimizer_class(
        model.parameters(),
        **config["optimizer_params"],
    )
    scheduler_type = config.get("scheduler_type", None)
    if scheduler_type is not None:
        scheduler_class = getattr(
            nntts.schedulers,
            scheduler_type
        )
        scheduler = scheduler_class(
            optimizer=optimizer,
            **config["scheduler_params"]
        )
    else:
        scheduler = None
    if args.distributed:
        # wrap model for distributed training
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
        )
    logging.info(model)

    # define trainer
    trainer_class = getattr(
        nntts.trainers,
        config.get("trainer_type", "EfficientTTSTrainer")
    )
    trainer = trainer_class(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        # print("Start training")
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl"))
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
