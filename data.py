import os
import os.path
import six
import random
import numpy as np
from PIL import Image

import torch.utils.data as data
from torchvision import transforms

# import utils
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


sets_root = '/data0/datasets/imgcom'
training_sets = [
    'flickr_sobel150_canny50_size256_stride256',
    'unsplash_sobel150_canny50_size256_stride256',
    'quanjing_sobel150_canny50_size256_stride256',
]
    
testing_sets = [
    'kodak_size64_stride64',
]


def pre_transform(out_size):
    return transforms.Compose([
        transforms.RandomCrop((out_size, out_size)),
        transforms.RandomHorizontalFlip(),
        RandomVerticalFlip(),
        
    ])


def pre_transform_test(out_size):
    return transforms.Compose([
        transforms.RandomCrop((out_size, out_size)),
        # transforms.RandomHorizontalFlip(),
        # RandomVerticalFlip(),
    ])


def input_transform(crop_params):
    return transforms.Compose([
        #ZeroOut(crop_params),
        transforms.ToTensor(),
    ])


def input_transform_norm(out_size):
    return transforms.Compose([
        #ZeroOut(crop_params),
        transforms.Scale(out_size/2), 
        transforms.Scale(out_size), 
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.408, 0.458, 0.485]),  # VGG mean
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])


def target_transform_norm(out_size):
    return transforms.Compose([
        #RightBottomCrop(crop_params),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert("RGB")
    return img


class RandomVerticalFlip(object):
    """
    Vertically flip the given PIL.Image randomly with a probability of 0.5.
    From: transforms.RandomHorizontalFlip
    """
    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomFlipOrRotation(object):
    """
    Vertically flip the given PIL.Image randomly with a probability of 0.5.
    From: transforms.RandomHorizontalFlip
    """
    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        random_num = random.random()
        if random_num >= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            random_num -= 0.5
        if random_num < 0.125:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        elif random_num < 0.25:
            return img.transpose(Image.ROTATE_90)
        elif random < 0.375:
            return img.transpose(Image.ROTATE_270)
        return img


class ZeroOut(object):
    """
    Zero out the right & bottom pixels of the given PIL.Image.
    """
    def __init__(self, crop_params):
        self.crop_start = crop_params[0]
        self.crop_end = crop_params[1]

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be zeroed.

        Returns:
            np.ndarray: shape as input.
        """
        img_np = np.array(img)
        img_np[self.crop_start:self.crop_end, self.crop_start:self.crop_end, :] = 0
        return img_np


class RightBottomCrop(object):
    """
    Zero out the right & bottom pixels of the given PIL.Image.
    """
    def __init__(self, crop_params):
        # crop_params: (32, 64)
        self.crop_start = crop_params[0]
        self.crop_end = crop_params[1]

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            np.ndarray: shape as (crop_end - crop_start, crop_end - crop_start).
        """
        img_np = np.array(img)
        return img_np[self.crop_start:self.crop_end, self.crop_start:self.crop_end, :]


class LMDBDataset(data.Dataset):
    def __init__(self, db_path, cache_dir, pre_transform=None, input_transform_norm=None, target_transform_norm=None):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = os.path.join(cache_dir, '_cache_' + db_path.replace('/', '_'))
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))
        # Custom
        self.pre_transform = pre_transform
        self.input_transform_norm = input_transform_norm
        self.target_transform_norm = target_transform_norm
        #self.input_norm = input_norm

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        # Custom
        pre_img = self.pre_transform(img)
        input, target = pre_img, pre_img
        input = self.input_transform_norm(input)       
        #if self.target_transform:
        target = self.target_transform_norm(target)
        return input, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def get_dataset(out_size, cache_dir, mode):
    datasets_list = []
    if mode == 'train':
        sets_path = [os.path.join(sets_root, sets) for sets in training_sets]
    else:
        sets_path = [os.path.join(sets_root, sets) for sets in testing_sets]
    for set_path in sets_path:
        sub_set_names = os.listdir(set_path)
        for sub_set_name in sub_set_names:
            sub_set_path = os.path.join(set_path, sub_set_name)
            if (os.path.isdir(sub_set_path)) and ('_lmdb' in sub_set_name):
                if mode == 'train':
                    datasets_list.append(
                        LMDBDataset(
                            db_path=sub_set_path,
                            cache_dir=cache_dir,
                            pre_transform=pre_transform(out_size),
                            input_transform_norm=input_transform_norm(out_size),
                            target_transform_norm=target_transform_norm(out_size)
                        )
                    )
                else:
                    datasets_list.append(
                        LMDBDataset(
                            db_path=sub_set_path,
                            cache_dir=cache_dir,
                            pre_transform=pre_transform_test(out_size),
                            input_transform_norm=input_transform_norm(out_size),
                            target_transform_norm=target_transform_norm(out_size)
                        )
                    )
    return data.ConcatDataset(datasets_list)


# if __name__ == '__main__':
#     dataset = get_dataset(32, 'caches', mode='train')
#     print len(dataset)
