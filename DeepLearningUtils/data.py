import torch
import matplotlib.gridspec as grsp
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset
from glob import glob

import random



class IXI_Dataset_Grayvalues(Dataset):
    def __init__(self, path, num_samples=0, mode="train", seq="all", hospital="all", k=4):
        # colltect the pathes:
        img_names_t1=[]
        img_names_t2=[]
        if hospital=="all":
            img_names_t1 = glob(os.path.join(path, "IXI*T1_affine.png"))
            img_names_t2 = glob(os.path.join(path, "IXI*T2_affine.png"))
        if "guys" in hospital:
            img_names_t1 += glob(os.path.join(path, "IXI*-Guys-*T1_affine.png"))
            img_names_t2 += glob(os.path.join(path, "IXI*-Guys-*T2_affine.png"))
        if "hh" in hospital:
            img_names_t1 += glob(os.path.join(path, "IXI*-HH-*T1_affine.png"))
            img_names_t2 += glob(os.path.join(path, "IXI*-HH-*T2_affine.png"))
        if "iop" in hospital:
            img_names_t1 += glob(os.path.join(path, "IXI*-IOP-*T1_affine.png"))
            img_names_t2 += glob(os.path.join(path, "IXI*-IOP-*T2_affine.png"))
        self.images = []
        img_names_t1.sort()
        img_names_t2.sort()
        self.img_paths = []
        if mode == "train":
            for i in range(img_names_t1.__len__()):
                if i % 4 != k:
                    if seq == "all":
                        self.img_paths.append(img_names_t1[i])
                        self.img_paths.append(img_names_t2[i])
                    elif seq=="t1":
                        self.img_paths.append(img_names_t1[i])
                    elif seq=="t2":
                        self.img_paths.append(img_names_t2[i])
            random.shuffle(self.img_paths)
            if num_samples > 0:
                self.img_paths = self.img_paths[0:num_samples]
        else:
            for i in range(img_names_t1.__len__()):
                if i % 4 == k:
                    if seq == "all":
                        self.img_paths.append(img_names_t1[i])
                        self.img_paths.append(img_names_t2[i])
                    elif seq == "t1":
                        self.img_paths.append(img_names_t1[i])
                    elif seq == "t2":
                        self.img_paths.append(img_names_t2[i])
            if num_samples > 0:
                self.img_paths = self.img_paths[0:num_samples]
        for path in self.img_paths:
            img = np.array(Image.open(path).resize((192,256)))/255.
            self.images.append(torch.tensor(img).unsqueeze(0).float())

    def __len__(self):
        return self.img_paths.__len__()

    def __getitem__(self, item):
        return self.images[item], self.img_paths[item]




def generate_imges(num_imgs, model_test, latent_dim, path, train_data=None):
    device = next(model_test.parameters()).device
    plt.figure(figsize=(4 * num_imgs, 4))
    G = grsp.GridSpec(1, num_imgs)
    if not train_data is None:
        train_data=train_data.to(device)
        encoded = model_test.encode(train_data)
        mu = torch.mean(encoded, dim=0).unsqueeze(0)
        std = torch.std(encoded, dim=0).unsqueeze(0)
    else:
        mu=torch.zeros((1,latent_dim))
        std = torch.ones((1, latent_dim))
    for i in range(0, num_imgs):
        z = torch.normal(mean=mu, std=std).to(device)
        gen_img = model_test.dec(z)
        img = np.array((gen_img.cpu().squeeze().detach()))
        ax = plt.subplot(G[0, i])
        for l in range(img.shape[0]):
            ax.imshow(img, cmap="gray")
            ax.axis("scaled")
        plt.axis('off')
    plt.savefig(path + '/gen_imgs.png')


def interpolate_images(img1, img2, model_test,path,dims=[64,512],steps=4,type="app"):
    if not (type=="app" or type=="shape"):
        print("Setting type to app as default")
        type == "app"
    enc1=model_test.encode(img1)
    enc2=model_test.encode(img2)
    plt.figure(dpi=300)
    G = grsp.GridSpec(1, (steps+3))
    ax0 = plt.subplot(G[0, 0])
    ax0.imshow(img1.cpu().squeeze().detach(), cmap="gray")
    ax0.axis('off')
    ax0.set_title("source")
    axend=plt.subplot(G[0, steps+2])
    axend.imshow(img2.cpu().squeeze().detach(), cmap="gray")
    axend.axis('off')
    axend.set_title("target")
    if type=="app":
        diff=(enc2[:, :dims[0]]-enc1[:,:dims[0]])/steps
    elif type=="shape":
        diff = (enc2[:, dims[0]:] - enc1[:, dims[0]:]) / steps
    interpolated=enc1.clone()
    for i in range(steps+1):
        if type=="app":
            interpolated[:,:dims[0]]=enc1[:,:dims[0]]+(i)*diff
        elif type=="shape":
            interpolated[:, dims[0]:] = enc1[:, dims[0]:] + (i) * diff
        interpolated_img=model_test.dec(interpolated)
        ax = plt.subplot(G[0, i+1])
        ax.imshow(interpolated_img.cpu().squeeze().detach(), cmap="gray")
        ax.axis("scaled")
        ax.axis('off')
        if i==0:
            ax.set_title("rec.")
        elif i==steps:
            ax.set_title("rec.")
        else:
            ax.set_title("step"+str(i))
    plt.axis('off')
    plt.savefig(path + '/'+type+'_interpolate.png')
    plt.close()
    plt.figure(dpi=300)

