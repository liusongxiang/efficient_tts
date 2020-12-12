#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Setup Neural Network based Text-to-Speech Synthesis (NNTTS) libarary."""

import os
import pip
import sys

from distutils.version import LooseVersion
from setuptools import find_packages
from setuptools import setup
from importlib.machinery import SourceFileLoader


version = SourceFileLoader('nntts.version', 'nntts/version.py').load_module().version

if LooseVersion(sys.version) < LooseVersion("3.6"):
    raise RuntimeError(
        "This libarary requires Python>=3.6, "
        "but your Python is {}".format(sys.version))
if LooseVersion(pip.__version__) < LooseVersion("19"):
    raise RuntimeError(
        "pip>=19.0.0 is required, but your pip is {}. "
        "Try again after \"pip install -U pip\"".format(pip.__version__))

packages = find_packages(include=["nntts*"])
if os.path.exists("README.md"):
    with open("README.md", 'r', encoding="UTF-8") as fh:
        LONG_DESC = LONG_DESC = fh.read()
else:
    LONG_DESC = ""

setup(
    name='nntts',
    version=version,
    url="http://github.com/liusongxiang",
    author="Songxiang Liu",
    author_email="songxiangliu.cuhk@gmail.com",
    description="Neural network based TTS libarary.",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    packages=packages,
    license="MIT License",
    install_requires=[
        "torch>=1.5.0",
        "setuptools>=38.5.1",
        "librosa>=0.8.0",
        "soundfile>=0.10.2",
        "tensorboardX>=1.8",
        "matplotlib>=3.1.0",
        "PyYAML>=3.12",
        "tqdm>=4.26.1",
        "typeguard",
    ],
    extras_requires={
        "test": [
        ],
    },
    entry_points={
        "console_scripts": [
            "nntts-train=nntts.bin.train:main",
            "nntts-inference=nntts.bin.inference:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
