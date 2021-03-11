import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math


def gaussian(window_size, sigma, dim):
    window_size = [window_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32)
            for size in window_size
        ]
    )
    for size, std, mgrid in zip(window_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                  torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    return kernel / torch.sum(kernel)


def create_window(window_size, channel, dim):
    window = gaussian(window_size, 0.5,dim).contiguous()

    window = window.view(1, 1, *window.size())
    window = window.repeat(channel, *[1] * (window.dim() - 1))

    return window


def _ssim(img1, img2, window, window_size, channel, dim,size_average=True):
    if dim==3:
        mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)
    else:
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)


    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    if dim == 3:
        sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    else:
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None#create_window(window_size, self.channel,dim)

    def forward(self, img1, img2):
        channel = img1.shape[1]
        dim=len(img1.shape)-2
        window = create_window(self.window_size, channel,dim)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        self.window = window
        self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel,dim, self.size_average)


def ssim(img1, img2, window_size=15, size_average=True):
    channel = img1.shape[1]
    dim = len(img1.shape) - 2
    window = create_window(window_size, channel,dim)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel,dim, size_average)
