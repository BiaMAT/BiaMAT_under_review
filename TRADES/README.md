# Biased Multi-domain Adversarial Training

This repository is the official implementation of Biased Multi-domain Adversarial Training (BiaMAT).  
The implementation is based on the official code of TRADES (https://github.com/yaodongyu/TRADES).

## Prerequisites

* Python 3
* Pytorch (0.4.1)
* Numpy
* CUDA
* tqdm

## Training

To train the BiaMAT model in the paper, run this command:

```train
python train_BiaMAT.py --model-dir <name> --data-dir <path to dir containing cifar-10-batches-py> --aux-dataset-dir <path to Imagenet32_train>
```

## Evaluation

To evaluate the model on CIFAR-10, run:

```eval
python pgd_attack_cifar10.py --model-path <model path> --num-steps 100 --BiaMAT --random --attack-method <cw or pgd>
```
