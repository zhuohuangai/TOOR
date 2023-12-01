import numpy as np
import os

class CROSSSET:
    def __init__(self, root, split="l_train"):
        self.dataset = np.load(os.path.join(root, "crossset", split+".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        # index = self.dataset["index"][idx]
        return image, label, idx

    def __len__(self):
        return len(self.dataset["images"])