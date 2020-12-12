#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import time

import numpy as np
import librosa
import soundfile as sf
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from tqdm import tqdm

import nntts.models
from nntts.vocoders.hifigan_model import load_hifigan_generator
from nntts.text import text_to_sequence


def run_tts(args):
    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    with open(args.test_fid_scp, 'r') as f:
        test_id2wavpath_list = [l.strip().split("|") for l in f if l.strip()]
    logging.info(f"The number of features to be decoded = {len(test_id2wavpath_list)}.")
    
    # setup model
    if torch.cuda.is_available() and not args.use_cpu:
        logging.info("Using GPU for inference.")
        device = torch.device("cuda")
    else:
        logging.info("Using CPU for inference.")
        device = torch.device("cpu")
    step = args.checkpoint.split('-')[-1][:-4]
    # define models 
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
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict["model"])
    try:
        model.remove_weight_norm()
    except:
        pass
    model = model.eval().to(device)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    voc_model = load_hifigan_generator(device)
    
    if config["dataset_params"]["use_phnseq"]:
        with open(config["dataset_params"]["phnset_path"], 'r') as f:
            phn_list = [l.strip() for l in f]
        phn2idx = dict(zip(phn_list, range(len(phn_list))))
    else:
        phn2idx = None

    # start generation
    total_rtf = 0.0
    cnt = 0
    with torch.no_grad(), tqdm(test_id2wavpath_list[:10], desc="[decode]") as pbar:
        for idx, (wave_path, text) in enumerate(pbar, 1):
            fid = wave_path.split('/')[1][:-4]
            start = time.time()
            if config["dataset_params"]["use_phnseq"]:
                text_tensor = torch.LongTensor(
                    [phn2idx[p] for p in text.split()]).unsqueeze(0).to(device)
            else:
                text_tensor = torch.LongTensor(
                    text_to_sequence(text, ["english_cleaners"])).unsqueeze(0).to(device)
            with torch.no_grad():
                mel_pred, alpha = model.inference(text_tensor)
                y = voc_model(mel_pred.transpose(1, 2))
                y = y.squeeze()
            rtf = (time.time() - start) / (len(y) / 22050.)
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf
            
            mel_pred = mel_pred.squeeze().cpu().numpy().T
            alpha = alpha.squeeze().cpu().numpy()
            fig, ax = plt.subplots(2)
            ax[0].imshow(alpha)
            ax[1].imshow(mel_pred)
            fig.savefig(os.path.join(config["outdir"], f"{fid}_{step}.png"))

            # save as PCM 16 bit wav file
            sf.write(os.path.join(config["outdir"], f"{fid}_{step}.wav"),
                     y.cpu().numpy(), 22050, "PCM_16")
            cnt += 1

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / cnt:.05f}).")


def get_parser():
    parser = argparse.ArgumentParser(description="Singing Voice conversion")
    parser.add_argument(
        "--test_fid_scp",
        type=str,
        default="",
        help="Utterance file id list scp file path, have the following form:"
        " wav_path|wave-path-1"
        " wav_path|wave-path-2",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Use CPU to inference.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="yaml format configuratio file. if not explicitly provided,"
        "it will be searched in the checkpoint directory. (default=None)"
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        required=True,
        help="directory to save generated speech."
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="checkpoint file to be loaded."
    )
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=1,
        help="logging level. higher is more logging. (default=1)"
    )
    return parser


def main():
    """Run decoding process."""
    parser = get_parser()
    args = parser.parse_args()
    run_tts(args)

if __name__ == "__main__":
    main()
