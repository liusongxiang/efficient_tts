# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

import fnmatch
import logging
import os
import sys
import tarfile

from distutils.version import LooseVersion

import numpy as np
import torch
import yaml

def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


class NpyScpLoader(object):
    """Loader class for a fests.scp file of npy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...
        >>> loader = NpyScpLoader("feats.scp")
        >>> array = loader["key1"]

    """

    def __init__(self, feats_scp):
        """Initialize npy scp loader.

        Args:
            feats_scp (str): Kaldi-style feats.scp file with npy format.

        """
        with open(feats_scp) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        self.data = {}
        for line in lines:
            key, value = line.split()
            self.data[key] = value

    def get_path(self, key):
        """Get npy file path for a given key."""
        return self.data[key]

    def __getitem__(self, key):
        """Get ndarray for a given key."""
        return np.load(self.data[key])

    def __len__(self):
        """Return the length of the scp file."""
        return len(self.data)

    def __iter__(self):
        """Return the iterator of the scp file."""
        return iter(self.data)

    def keys(self):
        """Return the keys of the scp file."""
        return self.data.keys()

    def values(self):
        """Return the values of the scp file."""
        for key in self.keys():
            yield self[key]


def load_model(checkpoint, config=None):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.

    Return:
        torch.nn.Module: Model instance.

    """
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    import parallel_wavegan.models

    # get model and load parameters
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator")
    )
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )

    # add pqmf if needed
    if config["generator_params"]["out_channels"] > 1:
        # lazy load for circular error
        from parallel_wavegan.layers import PQMF

        pqmf_params = {}
        # if LooseVersion(config.get("version", "0.1.0")) <= LooseVersion("0.4.2"):
            # For compatibility, here we set default values in version <= 0.4.2
            # pqmf_params.update(taps=62, cutoff_ratio=0.15, beta=9.0)
        model.pqmf = PQMF(
            subbands=config["generator_params"]["out_channels"],
            **config.get("pqmf_params", pqmf_params),
        )

    return model


def download_pretrained_model(tag, download_dir=None):
    """Download pretrained model form google drive.

    Args:
        tag (str): Pretrained model tag.
        download_dir (str): Directory to save downloaded files.

    Returns:
        str: Path of downloaded model checkpoint.

    """
    assert tag in PRETRAINED_MODEL_LIST, f"{tag} does not exists."
    id_ = PRETRAINED_MODEL_LIST[tag]
    if download_dir is None:
        download_dir = os.path.expanduser("~/.cache/parallel_wavegan")
    output_path = f"{download_dir}/{tag}.tar.gz"
    os.makedirs(f"{download_dir}", exist_ok=True)
    if not os.path.exists(output_path):
        # lazy load for compatibility
        import gdown

        gdown.download(f"https://drive.google.com/uc?id={id_}", output_path, quiet=False)
        with tarfile.open(output_path, 'r:*') as tar:
            for member in tar.getmembers():
                if member.isreg():
                    member.name = os.path.basename(member.name)
                    tar.extract(member, f"{download_dir}/{tag}")
    checkpoint_path = find_files(f"{download_dir}/{tag}", "checkpoint*.pkl")

    return checkpoint_path[0]
