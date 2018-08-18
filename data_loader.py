import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

PIX2PIX_DATASETS = [
    'facades', 'cityscapes', 'maps', 'edges2shoes', 'edges2handbags']

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, scale_size):
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))

        self.name = os.path.basename(root)
        #if self.name in PIX2PIX_DATASETS and not skip_pix2pix_processing:
        #    pix2pix_split_images(self.root)

        self.paths = glob(os.path.join(self.root, '*'))
        if len(self.paths) == 0:
            raise Exception("No images are found in {}".format(self.root))
        self.shape = list(Image.open(self.paths[0]).size) + [3]

        self.transform = transforms.Compose([
            #transforms.Scale(512),  #scale_size for dataset with images of different sizes
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.paths)

def get_loader(root, batch_size, scale_size, num_workers=2, shuffle=True):
    a_data_set = Dataset(root, scale_size)
    a_data_loader = torch.utils.data.DataLoader(dataset=a_data_set,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)
    a_data_loader.shape = a_data_set.shape

    return a_data_loader

