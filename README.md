# SGEN-pytorch
PyTorch implementation of "Sequential Gating Ensemble Network for Noise Robust Multi-Scale Face Restoration"
## Requirements

- Python 2.7
- [PyTorch](https://github.com/pytorch/pytorch)
- [torch-vision](https://github.com/pytorch/vision)


## Dataset

You first need to download the CelebA dataset from [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (you're looking for a file called img_align_celeba.zip). Then, you need to create a folder structure as data/dataset_name/1.jpg,...,2.jpg,...

## Training without GAN:

    $ python main.py --dataset=dataset_name 
    For example:
    $ python main.py --dataset=img_align_celeba

## Training with GAN:

    $ python main.py --dataset img_align_celeba --is_trainwithGAN True
## Testing:

    $ python main.py --is_train False --dataset img_align_celeba_test --load_path logs/img_align_celeba_2018-08-13_13-50-12


