import os
from PIL import Image
from torch import utils


class FolderDataset(utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.samples = os.listdir(self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        img = Image.open(os.path.join((self.root, sample)))
        return img



