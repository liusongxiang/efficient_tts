# EfficientTTS
## Unofficial Pytorch implementation of "EfficientTTS: An Efficient and High-Quality Text-to-Speech Architecture"([arXiv](https://arxiv.org/abs/2012.03500)).
### Disclaimer: Somebody mistakenly think I'm one of the authors. In fact, I am not even in the author list of this paper. I am just a TTS enthusiast. Some important information of the implementation is not presented by the paper. Some model parameters in current version is based on my understanding and exepriments, which may not be consistent with those used by the authors.



## Updates
2020/12/20: Using the HifiGAN finetuned with Tacotron2 GTA mel spectrograms can increase the quality of the generated samples, please see the newly [generated-samples](https://github.com/liusongxiang/efficient_tts/tree/main/egs/lj/checkpoint-320000steps)

## Current status
- [x] Implementation of EFTS-CNN + HifiGAN

## Setup with virtualenv

```
$ cd tools
$ make
# If you want to use distributed training, please run following
# command to install apex.
$ make apex
```

Note: If you want to specify Python version, CUDA version or PyTorch version, please run for example:

```
$ make PYTHON=3.7 CUDA_VERSION=10.1 PYTORCH_VERSION=1.6
```

## Training
Please go to egs/lj folder, and see run.sh for example use.

## Acknowledgement
The code framework is from https://github.com/kan-bayashi/ParallelWaveGAN


