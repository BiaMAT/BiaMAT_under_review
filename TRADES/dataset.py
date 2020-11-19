from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class ImageNet32(Dataset):
    def __init__(self, img, target, transform=None):
        self.len = img.shape[0]
        self.img = img
        self.target = target
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        img = (Image.fromarray((self.img[index]).astype(np.uint8)))

        if self.transform is not None:
            img = self.transform(img)
        return img, self.target[index]
