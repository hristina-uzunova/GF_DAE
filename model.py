import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions
from GF_DAE_Code.DeepLearningUtils.SSIM import SSIM

import math
from GF_DAE_Code.GuidedFilter.guided_filter import GuidedFilter
import matplotlib.pyplot as plt
import numpy as np


############## Variables #############
smallest_img_dim_y=24
smallest_img_dim_x= 32

def set_smallest(x=24, y=32):
    global smallest_img_dim_x
    global smallest_img_dim_y
    smallest_img_dim_y=y
    smallest_img_dim_x=x

def compute_grid(image_size, dtype=torch.float32, device=torch.device("cuda:0")):

    num_imgs = image_size[0]
    nx = image_size[1]
    ny = image_size[2]

    x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
    y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)

    x = x.expand(ny, -1).transpose(0, 1)
    y = y.expand(nx, -1)


    x.unsqueeze_(0).unsqueeze_(3)
    y.unsqueeze_(0).unsqueeze_(3)

    grid = torch.cat((y, x), 3).to(dtype=dtype, device=device)
    grid_list=[]
    for i in range(num_imgs):
        grid_list.append(grid)
    grid = torch.cat(grid_list, dim=0)

    return grid


def diffeomorphic_2D(displacement, grid, scaling=-1):

    scaling=8

    displacement = displacement / (2 ** scaling)
    displacement = displacement.transpose(3, 2).transpose(2, 1)
    for i in range(scaling):
        displacement_trans = displacement.transpose(1, 2).transpose(2, 3)
        displacement = displacement + F.grid_sample(displacement, displacement_trans + grid)

    return displacement.transpose(1, 2).transpose(2, 3)
def warp_image(image, displacement, mode="bilinear"):

    image_size = []
    image_size.append(image.size(0))
    image_size.append(image.size(2))
    image_size.append(image.size(3))

    grid = compute_grid(image_size, dtype=image.dtype, device=image.device)
    displacement = diffeomorphic_2D(displacement, grid, scaling=-1)
    # warp image
    warped_image = F.grid_sample(image, displacement + grid, mode=mode,align_corners=True)

    return warped_image


def reconstruct_img(templ_img, displ_field_fun, img_dim):
    num_imges = int(len(displ_field_fun)/1)

    template_img_batch = torch.Tensor(templ_img.cpu()).expand(num_imges, 1, img_dim[0], img_dim[1]).to(templ_img.device)
    template_img_batch = template_img_batch.reshape(-1, 1, img_dim[0], img_dim[1])

    rec_img_batch = warp_image(template_img_batch, displ_field_fun)

    rec_img_batch = rec_img_batch.view(num_imges, 1, img_dim[0], img_dim[1])

    return rec_img_batch


def loss_function(recon_img, input_img, disp_field, mu, logvar, diff_fac=10, loss_type="L1"):

    if loss_type=="L1":
        rec_func = nn.L1Loss(reduction='mean')
    elif loss_type=="SSIM":
        rec_func=SSIM()
    else:
        rec_func = nn.MSELoss(reduction='mean')

    in_out_diff = rec_func(recon_img, input_img)
    if loss_type == "SSIM":
        in_out_diff=1-in_out_diff
    if logvar is not None:
        kld = 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1. - logvar)
    else:
        kld=0

    ### DiffusionReguliser ###
    dx = (disp_field[:, 1:, 1:, :] - disp_field[:, :-1, 1:, :]).pow(2)
    dy = (disp_field[:, 1:, 1:, :] - disp_field[:, 1:, :-1, :]).pow(2)

    d_disp_field = F.pad(dx + dy, (0, 0, 0, 1, 0, 1)).mean()+torch.mean(torch.abs(disp_field))
    return in_out_diff + kld + diff_fac * d_disp_field


class ADAE(nn.Module):
    def __init__(self, latent_dim_a, latent_dim_s, img_dim,template, smooth=False, fixed=False, variational=True):
        super().__init__()
        self.latent_dim_a = latent_dim_a
        self.latent_dim_s = latent_dim_s
        if fixed:
            self.template=template
        else:
            self.template = nn.Parameter(template)
        self.enc = Enoder(latent_dim_a+latent_dim_s)
        self.dec_a = DecoderA(latent_dim_a, img_dim)
        self.dec_s = DecoderS(latent_dim_s, img_dim)
        self.smooth=smooth
        self.f = GuidedFilter(4).to(template.device)
        self.variational = variational
    def reparam(self, mu, logvar, training):
        if training:
            std = logvar.mul(0.5).exp_()
            device=std.device
            eps = torch.FloatTensor(std.size()).normal_().to(device)
            z = eps * std + mu
            return z
        else:
            return mu
    def encode(self, x):
        mu, logvar = self.enc(x)
        if self.variational:
            return self.reparam(mu, logvar, self.training)
        else:
           return mu

    def dec(self, z):
        self.diff_a = self.dec_a(z[:, 0:self.latent_dim_a])
        if self.smooth:
            self.diff_a = self.f(self.template.unsqueeze(0), self.diff_a)
        _, recon_img = self.dec_s(z[:, self.latent_dim_a:], template=self.template+self.diff_a)
        return recon_img

    def forward(self, x):
        mu, logvar = self.enc(x)
        if self.variational:
            z = self.reparam(mu, logvar, self.training)
        else:
            z=mu
            logvar=None
        self.diff_a=self.dec_a(z[:,0:self.latent_dim_a])
        if self.smooth:
            self.diff_a=self.f(self.template.unsqueeze(0),self.diff_a)
        displ_field, recon_img = self.dec_s(z[:,self.latent_dim_a:], template=self.template+self.diff_a)
        return displ_field, recon_img, mu, logvar


class Enoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv3 = nn.Conv2d(40, 80, kernel_size=2, stride=2, padding=0, bias=True)
        self.linear_1 = nn.Linear(smallest_img_dim_x * smallest_img_dim_y * 80, latent_dim)
        self.linear_2 = nn.Linear(smallest_img_dim_x * smallest_img_dim_y * 80, latent_dim)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))

        z_mu = self.linear_1(x.view(x.size(0), -1))
        z_var = self.linear_2(x.view(x.size(0), -1))

        return z_mu, z_var


class DecoderS(nn.Module):

    def __init__(self, latent_dim, img_dim):
        super().__init__()

        self.img_dim = img_dim
        self.linear = nn.Linear(latent_dim, smallest_img_dim_x * smallest_img_dim_y * 80)
        self.de_conv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                      nn.Conv2d(80, 40, kernel_size=3, stride=1, padding=1, bias=True))
        self.de_conv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                      nn.Conv2d(40, 20, kernel_size=3, stride=1, padding=1, bias=True))
        self.de_conv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                      nn.Conv2d(20, 2, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, x, template):
        x = torch.tanh(self.linear(x))
        x = torch.tanh(self.de_conv1(x.view(-1, 80, smallest_img_dim_x, smallest_img_dim_y)))
        x = torch.tanh(self.de_conv2(x))
        ret_displ_field = torch.tanh(self.de_conv3(x))
        ret_displ_field = ret_displ_field.permute(0, 2, 3, 1)
        recon_img = reconstruct_img(templ_img=template, displ_field_fun=ret_displ_field, img_dim=self.img_dim)
        return ret_displ_field, recon_img

class DecoderA(nn.Module):

    def __init__(self, latent_dim, img_dim):
        super().__init__()

        self.img_dim = img_dim
        self.linear = nn.Linear(latent_dim, smallest_img_dim_x * smallest_img_dim_y * 80)
        self.de_conv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                      nn.Conv2d(80, 40, kernel_size=3, stride=1, padding=1, bias=True))
        self.de_conv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                      nn.Conv2d(40, 20, kernel_size=3, stride=1, padding=1, bias=True))
        self.de_conv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"),
                                      nn.Conv2d(20, 1, kernel_size=3, stride=1, padding=1, bias=True))
    def forward(self, x):
        x = torch.tanh(self.linear(x))
        x = torch.tanh(self.de_conv1(x.view(-1, 80, smallest_img_dim_x, smallest_img_dim_y)))
        x = torch.tanh(self.de_conv2(x))
        x = self.de_conv3(x)
        return x

