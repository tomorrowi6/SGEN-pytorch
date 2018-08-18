
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from scipy import signal
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
import math
# imgs1/imgs2: 0 ~ 255, float32, numpy, rgb
# [batch_size, height, width, channel]
# return: list
def get_imgs_psnr(imgs1, imgs2):
    # per image
    def get_psnr(img1, img2):
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return 100
        pixel_max = 255.0
        # return 10 * math.log10((pixel_max ** 2) / mse)
        return 20 * math.log10(pixel_max / math.sqrt(mse))

    

    assert imgs1.shape[0] == imgs2.shape[0], "Batch size is not match."
    assert np.mean(imgs1) >= 1.0, "Check 1st input images range"
    assert np.mean(imgs2) >= 1.0, "Check 2nd input images range"
    PSNR_RGB= list()
    for num_of_batch in xrange(imgs1.shape[0]):
        PSNR_RGB.append(get_psnr(imgs1[num_of_batch], imgs2[num_of_batch]))

    return PSNR_RGB


def get_psnr(raw_imgs, recovered_imgs):
    PSNR_RGB= get_imgs_psnr(raw_imgs, recovered_imgs)
   
    return np.mean(PSNR_RGB)

