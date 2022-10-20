from collections import namedtuple
import glob
from cv2.ximgproc import guidedFilter
import sys
from net import *
from net.losses import StdLoss
from utils.imresize import imresize, np_imresize
from utils.image_io import *
from utils.file_io import write_log
from skimage.color import rgb2hsv
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import torch
import torch.nn as nn
from net.vae import VAE
import numpy as np
from net.Net import Net
from options import options


def get_dark_channel(image, w=15):
    """
    Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    image:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = image.shape
    padded = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(image, p=0.0001, w=15):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    image:      the 3 * M * N RGB image data ([0, L-1]) as numpy array
    w:      window for dark channel
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    image = image.transpose(1, 2, 0)
    # reference CVPR09, 4.4
    darkch = get_dark_channel(image, w)
    M, N = darkch.shape
    flatI = image.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)



def dehazing(opt):
    torch.cuda.set_device(opt.cuda)
    file_name = 'log/' + opt.datasets + '_' + opt.name + '.txt'

    if opt.datasets == 'SOTS':
        hazy_add = 'data/' + opt.datasets + '/synthetic/*.png'
        img_num = 500
    elif opt.datasets == 'HSTS':
        hazy_add = 'data/' + opt.datasets + '/synthetic/*.jpg'
        img_num = 10
    else:
        print('There are no proper datasets')
        return

    print(hazy_add, img_num)

    rec_psnr = 0
    rec_ssim = 0

    for item in sorted(glob.glob(hazy_add)):
        print(item)
        if opt.datasets == 'SOTS' or opt.datasets == 'HSTS':
            name = item.split('.')[0].split('/')[3]
        elif opt.datasets == 'real-world':
            name = item.split('.')[0].split('/')[2]
        print(name)

        if opt.datasets == 'SOTS':
            gt_add = 'data/' + opt.datasets + '/original/' + name.split('_')[0] + '.png'
        elif opt.datasets == 'HSTS':
            gt_add = 'data/' + opt.datasets + '/original/' + name + '.jpg'

        hazy_img = prepare_image(item)
        gt_img = prepare_gt(gt_add, dataset=opt.datasets)

        dh = Dehaze(name, hazy_img, gt_img, opt)
        dh.optimize()
        dh.finalize()
        psnr = dh.best_result.psnr
        ssim = dh.best_result_ssim.ssim

        write_log(file_name, name, psnr, ssim)

        rec_psnr += psnr
        rec_ssim += ssim

    rec_psnr = rec_psnr / img_num
    rec_ssim = rec_ssim / img_num
    write_log(file_name, 'Average', rec_psnr, rec_ssim)

if __name__ == "__main__":
    dehazing(options)
