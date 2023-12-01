from scipy.misc.doccer import indentcount_lines
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tiny_imagenet import TinyImageNet200
from imagenet32 import IMAGENET32
from torch.utils.data import Sampler
import numpy as np
import torch
import torchvision.transforms as tv_transforms
import os, argparse
import scipy.misc as misc

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="PI", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL]")
parser.add_argument("--em", default=0.2, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=500, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="crossset", type=str, help="dataset name : [crossset, svhn, cifar10, cifar100, mnist, imagenet32, tin]")
parser.add_argument("--n_labels", "-n", default=10000, type=int, help="the number of labeled data")
parser.add_argument("--n_unlabels", "-u", default=100000, type=int, help="the number of unlabeled data")
parser.add_argument('--n_valid', default=5000, type=int)
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="./exp_res", type=str, help="output dir")
parser.add_argument("--ratio", "-rat", default=0.5, type=float, help="ratio for class mismatch")
parser.add_argument("--threshold", "-th", default=0.9, type=float, help="threshold")
parser.add_argument("--gpus", default=1, type=int, help="number of GPUs") # using 1 GPUs.
args = parser.parse_args()

COUNTS = {
    "svhn": {"train": 73257, "test": 26032, "valid": 7326},
    "cifar10": {"train": 50000, "test": 10000, "valid": 5000},
    "cifar100": {"train": 50000, "test": 10000, "valid": 5000},
    "imagenet32": {"train": 1281167, "test": 50000, "valid": 50050},
    "imagenet32": {"train": 100000, "test": 10000, "valid": 5000},
}
rng = np.random.RandomState(seed=1)

_DATA_DIR = "./data"

if args.dataset == 'crossset':
    args.n_labels = 60
    args.n_unlabels = 20000
elif args.dataset == 'cifar100':
    args.n_labels = 6000
    args.n_unlabels = 20000
elif args.dataset == 'imagenet32':
    args.n_labels = 6000
    args.n_unlabels = 20000
elif args.dataset == 'tin':
    args.n_labels = 12000
    args.n_unlabels = 40000


def split_l_u(l_train_set, u_train_set, n_labels, n_unlabels):
    # NOTE: this function merges the two datasets and creates a class mismatch train_set.
    # cls number l_train_set is default 100, and 200 for u_train_set.
    # n_labels: 10k, n_unlabels: 500k (Laine et al. Temporal Ensembling).
    l_train_images = l_train_set["images"]
    l_train_labels = l_train_set["labels"]
    u_train_images = u_train_set["images"]
    u_train_labels = u_train_set["labels"]
    l_classes = np.unique(l_train_labels)
    u_classes = np.unique(u_train_labels)
    n_labels_per_cls = n_labels // len(l_classes)
    print("labeled per cls:", n_labels_per_cls)
    n_unlabels_per_cls = n_unlabels // len(u_classes)
    print("unlabeled per cls:", n_unlabels_per_cls)
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    for c in l_classes:
        cls_mask = (l_train_labels == c)
        c_images = l_train_images[cls_mask]
        c_labels = l_train_labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
    for c in u_classes:
        cls_mask = (u_train_labels == c)
        c_images = u_train_images[cls_mask]
        c_labels = u_train_labels[cls_mask]
        u_images += [c_images[:n_unlabels_per_cls]]
        u_labels += [c_labels[:n_unlabels_per_cls]]

    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}

    # shuffle
    indices = rng.permutation(len(l_train_set["images"]))
    l_train_set["images"] = l_train_set["images"][indices]
    l_train_set["labels"] = l_train_set["labels"][indices]

    indices = rng.permutation(len(u_train_set["images"]))
    u_train_set["images"] = u_train_set["images"][indices]
    u_train_set["labels"] = u_train_set["labels"][indices]
    return l_train_set, u_train_set
    
def reduce_classes(dataset, class_list):
    images = dataset["images"]
    labels = dataset['labels']
    classes = np.unique(labels)
    l_images = []
    l_labels = []
    for c in class_list:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:]]
        l_labels += [c_labels[:]]
    dataset = {"images" : np.concatenate(l_images, 0), "labels" : np.concatenate(l_labels,0)}

    indices = rng.permutation(len(dataset["images"]))
    dataset["images"] = dataset["images"][indices]
    dataset["labels"] = dataset["labels"][indices]
    return dataset

def split_test(test_set, tot_class=60):
    images = test_set["images"]
    labels = test_set['labels']
    classes = np.unique(labels)
    l_images = []
    l_labels = []
    for c in classes[:tot_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:]]
        l_labels += [c_labels[:]]
    test_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels,0)} # , "index": np.concatenate(idxs, 0)}

    indices = rng.permutation(len(test_set["images"]))
    test_set["images"] = test_set["images"][indices]
    test_set["labels"] = test_set["labels"][indices]
    return test_set

def load_cifar100():
    splits = {}
    for train in [True, False]:
        dataset = datasets.CIFAR100(_DATA_DIR, train, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits['train' if train else 'test'] = data
    return splits.values()

def load_imagenet32():
    splits = {}
    for train in [True, False]:
        dataset = IMAGENET32(_DATA_DIR, train, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits['train' if train else 'test'] = data
    return splits.values()

    
def load_mnist():
    splits = {}
    trans = tv_transforms.Compose([tv_transforms.ToPILImage(), tv_transforms.ToTensor(), tv_transforms.Normalize((0.5,), (1.0,))])
    for train in [True, False]:
        dataset = datasets.MNIST(_DATA_DIR, train, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits['train' if train else 'test'] = data
    return splits.values()


def load_svhn():
    splits = {}
    for split in ["train", "test", "extra"]:
        tv_data = datasets.SVHN(_DATA_DIR, split, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.labels
        splits[split] = data
    return splits.values()

def load_ImageNet32():
    splits = {}
    for train in [True, False]:
        dataset = TinyImageNet200(_DATA_DIR, train)
        data = {}
        data["images"] = dataset.data
        data["labels"] = np.array(dataset.target)
        splits["train" if train else "test"] = data
    return splits.values()


def gcn(images, multiplier=55, eps=1e-10):
    #global contrast normalization
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    # the following line is limited on the size of input images.
    # per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    per_image_norm[per_image_norm < eps] = 1
    images = multiplier * images / per_image_norm
    return images

def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp

def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)

rng = np.random.RandomState(seed=1)
tot_class = 6

if args.dataset == "crossset":
    train_set_svhn, test_set_svhn, _  = load_svhn()
    train_set_mnist, test_set_mnist = load_mnist()

    transform = False

train_set_svhn['images'] = train_set_svhn['images'][:args.n_labels]
train_set_svhn['labels'] = train_set_svhn['labels'][:args.n_labels]
train_set_mnist['images'] = train_set_mnist['images'][args.n_labels:(args.n_labels+args.n_unlabels)]
train_set_mnist['labels'] = train_set_mnist['labels'][args.n_labels:(args.n_labels+args.n_unlabels)]
test_set_svhn['images'] = train_set_svhn['images'][args.n_valid:]
test_set_svhn['labels'] = train_set_svhn['labels'][args.n_valid:]
indices = rng.permutation(len(train_set_svhn['labels']))
train_set_svhn['images'] = train_set_svhn['images'][indices]
train_set_svhn['labels'] = train_set_svhn['labels'][indices]
indices = rng.permutation(len(train_set_mnist['labels']))
train_set_mnist['images'] = train_set_mnist['images'][indices]
train_set_mnist['labels'] = train_set_mnist['labels'][indices]
indices = rng.permutation(len(test_set_svhn['labels']))
test_set_svhn['images'] = train_set_svhn['images'][indices]
test_set_svhn['labels'] = train_set_svhn['labels'][indices]

#split training set into training and validation
train_labeled_image = train_set_svhn['images']
train_labeled_label = train_set_svhn['labels']
train_unlabeled_image = train_set_mnist['images']
train_unlabeled_label = train_set_mnist['labels']
validation_images = test_set_svhn['images']
validation_labels = test_set_svhn['labels']
validation_set = {'images': validation_images, 'labels': validation_labels}
l_train_set = {'images': train_labeled_image, 'labels': train_labeled_label}
u_train_set = {'images': train_unlabeled_image, 'labels': train_unlabeled_label}

# if not os.path.exists(os.path.join(_DATA_DIR, args.dataset)):
#     os.mkdir(os.path.join(_DATA_DIR, args.dataset))

# np.save(os.path.join(_DATA_DIR, args.dataset, "l_train"), l_train_set)
# np.save(os.path.join(_DATA_DIR, args.dataset, "u_train"), u_train_set)
# # np.save(os.path.join(_DATA_DIR, args.dataset, "val"), validation_set)
# np.save(os.path.join(_DATA_DIR, args.dataset, "test"), validation_set)