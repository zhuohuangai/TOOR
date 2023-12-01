from lib.datasets import svhn, cifar10, cifar100, crossset, imagenet32, tin
import numpy as np

shared_config = {
    "iteration" : 100000,
    "warmup" : 40000,
    "lr_decay_iter" : 80000,
    "lr_decay_factor" : 0.2,
    "batch_size" : 100,
}
### dataset ###
svhn_config = {
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
}
cifar10_config = {
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,
}
cifar100_config = {
    "transform" : [True, True, True],
    "dataset" : cifar100.CIFAR100,
    "num_classes" : 60,
}
imagenet32_config = {
    "transform" : [True, True, True],
    "dataset" : imagenet32.IMAGENET32,
    "num_classes" : 60,
}
tin_config = {
    "transform" : [True, True, True],
    "dataset" : tin.TIN,
    "num_classes" : 120,
}
crossset_config = {
    "transform" : [True, True, True],
    "dataset" : crossset.CROSSSET,
    "num_classes" : 60,
}

### algorithm ###
vat_config = {
    # virtual adversarial training
    "xi" : 1e-6,
    "eps" : {"cifar10":6, "svhn":1, "crossset":6, "cifar100":6, "imagenet32":6, "tinyimagenet":6},
    "consis_coef" : 0.3,
    "lr" : 3e-3
}
pl_config = {
    # pseudo label
    "threashold" : 0.95,
    "lr" : 3e-4,
    "consis_coef" : 1,
}
mt_config = {
    # mean teacher
    "ema_factor" : 0.95,
    "lr" : 4e-4,
    "consis_coef" : 8,
}
pi_config = {
    # Pi Model
    "lr" : 3e-4,
    "consis_coef" : 20.0,
}
te_config = {
    # interpolation consistency training
    "ema_factor" : 0.999,
    "lr" : 3e-4,
    "consis_coef" : 20.0,
}
supervised_config = {
    "lr" : 3e-3
}
### master ###
config = {
    "shared" : shared_config,
    "svhn" : svhn_config,
    "cifar10" : cifar10_config,
    "cifar100" : cifar100_config,
    "imagenet32" : imagenet32_config,
    "tin" : tin_config,
    "crossset" : crossset_config,
    "VAT" : vat_config,
    "PL" : pl_config,
    "MT" : mt_config,
    "PI" : pi_config,
    "TE" : te_config,
    "supervised" : supervised_config
}




