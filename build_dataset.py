from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tiny_imagenet import TinyImageNet200
from imagenet32 import IMAGENET32
from torch.utils.data import Sampler
import numpy as np
import torch
import torchvision.transforms as tv_transforms
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="PI", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL]")
parser.add_argument("--em", default=0.2, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=500, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="cifar100", type=str, help="dataset name : [crossset, svhn, cifar10, cifar100, mnist, imagenet32, tin]")
parser.add_argument("--n_labels", "-n", default=6000, type=int, help="the number of labeled data")
parser.add_argument("--n_unlabels", "-u", default=20000, type=int, help="the number of unlabeled data")
parser.add_argument('--n_valid', default=5000, type=int)
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="./exp_res", type=str, help="output dir")
parser.add_argument('--temper', default=0.8, type=float, help='temperature scaling')
parser.add_argument("--ratio", "-rat", default=0.5, type=float, help="ratio for class mismatch")
parser.add_argument("--threshold", "-th", default=0.9, type=float, help="threshold")
parser.add_argument("--gpus", default=1, type=int, help="number of GPUs") # using 2 GPUs.
args = parser.parse_args()

COUNTS = {
    "svhn": {"train": 73257, "test": 26032, "valid": 7326},
    "cifar10": {"train": 50000, "test": 10000, "valid": 5000},
    "cifar100": {"train":50000, "test": 10000, "valid": 5000},
    "imagenet32": {"train": 1281167, "test": 50000, "valid": 50050},
    "tinyimagenet": {"train": 100000, "test": 10000, "valid": 5000},
}

if args.dataset == 'crossset':
    args.n_labels = 6000
    args.n_unlabels = 200000
elif args.dataset == 'cifar100':
    args.n_labels = 6000
    args.n_unlabels = 20000
elif args.dataset == 'imagenet32':
    args.n_labels = 6000
    args.n_unlabels = 200000
elif args.dataset == 'tin':
    args.n_labels = 12000
    args.n_unlabels = 40000


rng = np.random.RandomState(seed=1)

_DATA_DIR = "./data"

class SimpleDataset(Dataset):
    def __init__(self, dataset, transform=True):
        self.dataset=dataset
        self.transform=transform

    def __getitem__(self, index):
        image = self.dataset['images'][index]
        label = self.dataset['labels'][index]
        if(self.transform):
            image = (image / 255. - 0.5) / 0.5
        return image, label, index

    def __len__(self):
        return len(self.dataset['images'])

class RandomSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

data_path = "./data"

def split_l_u(train_set, n_labels, n_unlabels, tot_class=6, ratio=0.5):
    # NOTE: this function assume that train_set is shuffled.
    images = train_set["images"]
    labels = train_set["labels"]
    # idxs = np.arange(len(labels))
    classes = np.unique(labels)
    n_labels_per_cls = n_labels // tot_class
    n_unlabels_per_cls = int(n_unlabels*(1.0-ratio)) // tot_class
    if(tot_class < len(classes)):
        n_unlabels_shift = (n_unlabels - (n_unlabels_per_cls * tot_class)) // (len(classes) - tot_class)
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    for c in classes[:tot_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
        u_labels += [c_labels[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
    for c in classes[tot_class:]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        u_images += [c_images[:n_unlabels_shift]]
        u_labels += [c_labels[:n_unlabels_shift]]

    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}

    # shuffle
    indices = rng.permutation(len(l_train_set["images"]))
    l_train_set["images"] = l_train_set["images"][indices]
    l_train_set["labels"] = l_train_set["labels"][indices]
    # l_train_set["index"] = l_train_set["index"][indices]

    indices = rng.permutation(len(u_train_set["images"]))
    u_train_set["images"] = u_train_set["images"][indices]
    u_train_set["labels"] = u_train_set["labels"][indices]
    # u_train_set["index"] = u_train_set["index"][indices]
    return l_train_set, u_train_set

def split_test(test_set, tot_class=6):
    images = test_set["images"]
    labels = test_set['labels']
    # index = np.arange(len(labels))
    classes = np.unique(labels)
    l_images = []
    l_labels = []
    # idxs = []
    for c in classes[:tot_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        # c_idxs = index[cls_mask]
        l_images += [c_images[:]]
        l_labels += [c_labels[:]]
        # idxs += [c_idxs[:]]
    test_set = {"images": np.concatenate(l_images, 0), "labels":np.concatenate(l_labels,0)} # , "index": np.concatenate(idxs, 0)}

    indices = rng.permutation(len(test_set["images"]))
    test_set["images"] = test_set["images"][indices]
    test_set["labels"] = test_set["labels"][indices]
    # test_set["index"] = test_set["index"][indices]
    return test_set

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

def load_mnist():
    splits = {}
    trans = tv_transforms.Compose([tv_transforms.ToPILImage(), tv_transforms.ToTensor(), tv_transforms.Normalize((0.5,), (1.0,))])
    for train in [True, False]:
        dataset = datasets.MNIST(data_path, train, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits['train' if train else 'test'] = data
    return splits.values()


def load_svhn():
    splits = {}
    for split in ["train", "test", "extra"]:
        tv_data = datasets.SVHN(data_path, split, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.labels
        splits[split] = data
    return splits.values()

def load_cifar10():
    splits = {}
    for train in [True, False]:
        dataset = datasets.CIFAR10(data_path, train, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits["train" if train else "test"] = data
    return splits.values()


def load_cifar100():
    splits = {}
    for train in [True, False]:
        dataset = datasets.CIFAR100(data_path, train, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits["train" if train else "test"] = data
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

def load_TinyImageNet():
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
    images = images.astype(np.float64)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
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


# def get_dataloaders(dataset, n_labels, n_unlabels, n_valid, l_batch_size, ul_batch_size, test_batch_size, iterations,
#                     tot_class=6, ratio=0.5):
rng = np.random.RandomState(seed=1)
tot_class = 6

if args.dataset == "mnist":
    train_set, test_set = load_mnist()
    transform = False
elif args.dataset == "svhn":
    train_set, test_set, extra_set = load_svhn()
    transform = False
elif args.dataset == "cifar10":
    train_set, test_set = load_cifar10()
    train_set["images"] = gcn(train_set["images"])
    test_set["images"] = gcn(test_set["images"])
    mean, zca_decomp = get_zca_normalization_param(train_set["images"])
    train_set["images"] = zca_normalization(train_set["images"], mean, zca_decomp)
    test_set["images"] = zca_normalization(test_set["images"], mean, zca_decomp)
    # N x H x W x C -> N x C x H x W
    train_set["images"] = np.transpose(train_set["images"], (0, 3, 1, 2))
    test_set["images"] = np.transpose(test_set["images"], (0, 3, 1, 2))

    #move class "plane" and "car" to label 8 and 9
    train_set['labels'] -= 2
    test_set['labels'] -= 2
    train_set['labels'][np.where(train_set['labels'] == -2)] = 8
    train_set['labels'][np.where(train_set['labels'] == -1)] = 9
    test_set['labels'][np.where(test_set['labels'] == -2)] = 8
    test_set['labels'][np.where(test_set['labels'] == -1)] = 9
    transform = False
elif args.dataset == "cifar100":
    train_set, test_set = load_cifar100()
    train_set["images"] = gcn(train_set["images"])
    test_set["images"] = gcn(test_set["images"])
    mean, zca_decomp = get_zca_normalization_param(train_set["images"])
    train_set["images"] = zca_normalization(train_set["images"], mean, zca_decomp)
    test_set["images"] = zca_normalization(test_set["images"], mean, zca_decomp)
    # N x H x W x C -> N x C x H x W
    train_set["images"] = np.transpose(train_set["images"], (0, 3, 1, 2))
    test_set["images"] = np.transpose(test_set["images"], (0, 3, 1, 2))
    tot_class = 60
    transform = False
elif args.dataset == "imagenet32":
    train_set, test_set = load_imagenet32()
    fimnet32 = 'mapcifar2imnet.txt'
    mapcls = np.loadtxt(fimnet32, dtype=int)
    classes = np.unique(train_set['labels'])
    ood_classes = list(set(classes) - set(mapcls[:, 1]))[:40]
    class_list = np.concatenate([mapcls[:, 1], ood_classes], 0)
    label_map = {}
    for i in range(len(class_list)):
        label_map[class_list[i]] = i
    train_set = reduce_classes(train_set, class_list)
    test_set = reduce_classes(test_set, class_list)
    
    for i in range(len(train_set['labels'])):
        train_set['labels'][i] = label_map[train_set['labels'][i]]
    for i in range(len(test_set['labels'])):
        test_set['labels'][i] = label_map[test_set['labels'][i]]

    train_set["images"] = gcn(train_set["images"])
    test_set["images"] = gcn(test_set["images"])
    mean, zca_decomp = get_zca_normalization_param(train_set["images"])
    train_set["images"] = zca_normalization(train_set["images"], mean, zca_decomp)
    test_set["images"] = zca_normalization(test_set["images"], mean, zca_decomp)
    # N x H x W x C -> N x C x H x W
    train_set["images"] = np.transpose(train_set["images"], (0, 3, 1, 2))
    test_set["images"] = np.transpose(test_set["images"], (0, 3, 1, 2))
    
    tot_class = 60
    transform = False
elif args.dataset == "tin":
    train_set, test_set = load_TinyImageNet()
    train_set["images"] = gcn(train_set["images"])
    test_set["images"] = gcn(test_set["images"])
    mean, zca_decomp = get_zca_normalization_param(train_set["images"])
    train_set["images"] = zca_normalization(train_set["images"], mean, zca_decomp)
    test_set["images"] = zca_normalization(test_set["images"], mean, zca_decomp)
    # N x H x W x C -> N x C x H x W
    train_set["images"] = np.transpose(train_set["images"], (0, 3, 1, 2))
    test_set["images"] = np.transpose(test_set["images"], (0, 3, 1, 2))
    tot_class = 120
    transform = False

#permute index of training set
indices = rng.permutation(len(train_set['images']))
train_set['images'] = train_set['images'][indices]
train_set['labels'] = train_set['labels'][indices]

#split training set into training and validation
train_images = train_set['images'][args.n_valid:]
train_labels = train_set['labels'][args.n_valid:]
validation_images = train_set['images'][:args.n_valid]
validation_labels = train_set['labels'][:args.n_valid]
validation_set = {'images': validation_images, 'labels': validation_labels}
train_set = {'images': train_images, 'labels': train_labels}


#split training set into labeled and unlabeled data
validation_set = split_test(validation_set, tot_class=tot_class)
test_set = split_test(test_set, tot_class=tot_class)
l_train_set, u_train_set = split_l_u(train_set, args.n_labels, args.n_unlabels, tot_class=tot_class, ratio=args.ratio)

print("Unlabeled data in distribuiton : {}, Unlabeled data out distribution : {}".format(
        np.sum(u_train_set['labels'] < tot_class), np.sum(u_train_set['labels'] >= tot_class)))

if not os.path.exists(os.path.join(_DATA_DIR, args.dataset)):
    os.mkdir(os.path.join(_DATA_DIR, args.dataset))

np.save(os.path.join(_DATA_DIR, args.dataset, "l_train"), l_train_set)
np.save(os.path.join(_DATA_DIR, args.dataset, "u_train"), u_train_set)
np.save(os.path.join(_DATA_DIR, args.dataset, "val"), validation_set)
np.save(os.path.join(_DATA_DIR, args.dataset, "test"), test_set)