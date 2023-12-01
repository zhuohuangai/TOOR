from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class IMAGENET32(VisionDataset):
    """ImageNet 32x32
    """
    base_folder = 'Imagenet32'
    url = None
    filename = "Imagenet32_val.zip"
    train_list = [
        'train_data_batch_1',
        'train_data_batch_2',
        'train_data_batch_3',
        'train_data_batch_4',
        'train_data_batch_5',
        'train_data_batch_6',
        'train_data_batch_7',
        'train_data_batch_8',
        'train_data_batch_9',
        'train_data_batch_10',
    ]

    test_list = [
        'val_data',
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(IMAGENET32, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
            self.base_folder = self.base_folder + '_train'
        else:
            downloaded_list = self.test_list
            self.base_folder = self.base_folder + '_val'

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        # imagenet_32 labels vary from 1 to 1000.
        self.targets = [i-1 for i in self.targets]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

