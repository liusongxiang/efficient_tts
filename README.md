## Unofficial Pytorch implementation of "efficientTTS: An Efficient and High-Quality Text-to-Speech Architecture".

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


