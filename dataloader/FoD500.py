#! /usr/bin/python3

import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torch
from PIL import Image

from torchvision import transforms
import random
import numbers
import OpenEXR
from os import listdir, mkdir
from os.path import isfile, join, isdir
import cv2
from torchvision.transforms import functional as F

# code adopted from https://github.com/soyers/ddff-pytorch/blob/master/python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py

def read_dpt(img_dpt_path):
    # pt = Imath.PixelType(Imath.PixelType.HALF)  # FLOAT HALF
    dpt_img = OpenEXR.InputFile(img_dpt_path)
    dw = dpt_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    (r, g, b) = dpt_img.channels("RGB")
    dpt = np.fromstring(r, dtype=np.float16)
    dpt.shape = (size[1], size[0])
    return dpt
#[1, 1.5, 2.5, 4, 6]
class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""

    def __init__(self, root_dir, img_list, dpth_list,  transform_fnc=None, flag_shuffle=False, img_num=5, data_ratio=0,
                 flag_inputs=[True, False], flag_outputs=[False, True], focus_dist=[0.1,.15,.3,0.7,1.5], f_number=0.1, scale=1):
        self.root_dir = root_dir
        self.transform_fnc = transform_fnc
        self.flag_shuffle = flag_shuffle

        self.flag_rgb = flag_inputs[0]
        self.flag_coc = flag_inputs[1]

        self.img_num = img_num
        self.data_ratio = data_ratio

        self.flag_out_coc = flag_outputs[0]
        self.flag_out_depth = flag_outputs[1]

        self.focus_dist = focus_dist #torch.tensor(focus_dist) / scale
        self.max_n_stack = 5
        self.dpth_scale = scale


        self.img_mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3])#[0.278, 0.250, 0.227]
        self.img_std= np.array([0.229, 0.224, 0.225]).reshape([1,1,3])#[0.185, 0.178, 0.178]

        self.guassian_kernel =  (35,35) # large blur kernel for pad image

        ##### Load all images
        self.imglist_all = img_list
        self.imglist_dpt = dpth_list


    def __len__(self):
        return int(len(self.imglist_dpt))

    def dpth2disp(self, dpth):
        disp = 1 / dpth
        disp[dpth==0] = 0
        return disp

    def __getitem__(self, idx):
        ##### Read and process an image
        idx_dpt = int(idx)
        img_dpt = read_dpt(self.root_dir + self.imglist_dpt[idx_dpt])

        foc_dist = self.focus_dist.copy()
        mat_dpt = img_dpt.copy()[:, :, np.newaxis]/3

        img_num = min(self.max_n_stack, self.img_num)
        ind = idx * img_num

        num_list = list(range(self.max_n_stack))

        # add RGB, CoC, Depth inputs
        mats_input = []
        mats_output = np.zeros((256, 256, 0))

        # load existing image
        pad_lst = []
        pad_focs = []
        for i in range(self.max_n_stack):
            if self.flag_rgb:
                im = Image.open(self.root_dir + self.imglist_all[ind + num_list[i]])
                img_all = np.array(im)
                # img Norm
                mat_all = img_all.copy()/255
                # mat_all = (mat_all - self.img_mean) / self.img_std

                # mat_fd = foc_dist[i].view(1, 1, -1).expand(*mat_all.shape[:2],1)

                # mat_fd = np.array(foc_dist[i]/1.5).reshape(1, 1, 1)
                # mat_fd = np.tile(mat_fd, (mat_all.shape[0], mat_all.shape[1], 1))
                # mat_all = np.concatenate([mat_all, mat_fd], axis=2)     
                mats_input.append(mat_all)

                # pad invalid img in the beginning or the end, keep diff consistent
                if self.img_num > self.max_n_stack and len(pad_lst) == 0:
                    for j in range(self.img_num-self.max_n_stack):
                        pad_lst.append(cv2.GaussianBlur(mat_all, self.guassian_kernel, 0))
                        pad_focs.append(0)

        mats_input = pad_lst + mats_input
        foc_dist = pad_focs + foc_dist


        mats_input = np.stack(mats_input)
        if img_num < self.max_n_stack:
            if len( self.imglist_all) > 100: # train
                rand_idx = np.random.choice(self.max_n_stack, img_num,
                                            replace=False)  # this will shuffle order as well
                rand_idx = np.sort(rand_idx)
            else:
                rand_idx = np.linspace(0, self.max_n_stack, img_num)

            mats_input = mats_input[rand_idx]
            foc_dist = [foc_dist[i] for i in rand_idx]

        if self.flag_out_depth:
            mats_output = np.concatenate((mats_output,(mat_dpt) * self.dpth_scale), axis=2) # first 5 is COC last is depth  self.dpth2disp

        sample = {'input': mats_input, 'output': mats_output}

        if self.transform_fnc:
            sample = self.transform_fnc(sample)
        return sample['input'], sample['output'],(torch.tensor(foc_dist))#['input'], sample['output'],(torch.tensor(foc_dist)) * self.dpth_scale # to match the scale of DDFF12  self.dpth2disp

class ToTensor(object):
    def __call__(self, sample):
        mats_input, mats_output = sample['input'], sample['output']

        mats_input = mats_input.transpose((0, 3, 1, 2))
        mats_output = mats_output.transpose((2, 0, 1))
        # print(mats_input.shape, mats_output.shape)
        return {'input': torch.from_numpy(mats_input).float(),
                'output': torch.from_numpy(mats_output).float()}

class RandomRotate(object):
    """ Randomly rotate the image within a given range of degrees """
    def __init__(self, degrees):
        self.degrees = degrees  # degrees should be a tuple like (min, max)

    def __call__(self, sample):
        inputs, target = sample['input'], sample['output']
        
        # 随机选择旋转角度
        angle = random.randint(self.degrees[0], self.degrees[1])
        inputs = torch.rot90(inputs,angle,dims=(2,3))
        target = torch.rot90(target,angle,dims=(1,2))

        return {'input': inputs, 'output': target}

class RandomColorJitter(object):
    """ Randomly apply color jitter (contrast, brightness, and gamma) """
    def __init__(self, contrast_range=(0.5, 1.5), brightness_range=(-0.2, 0.2), gamma_range=(0.8, 1.2)):
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.gamma_range = gamma_range

    def __call__(self, sample):
        inputs, target = sample['input'], sample['output']
        # inputs=inputs/255
        # 随机选择对比度、亮度和伽马的调整范围
        contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
        brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
    
        # 只对通道数据之外的维度进行操作
        inputs[:, :-1, :, :] = (0.5 + contrast * (inputs[:, :-1, :, :] - 0.5)) + brightness
        inputs = torch.clamp(inputs, 0, 1)  # 将值限制在 [0, 1] 范围内
        
        # 对张量数据的最后一维通道之外的部分进行 gamma 变换
        inputs[:, :-1, :, :] = torch.pow(inputs[:, :-1, :, :], gamma)
        inputs = torch.clamp(inputs, 0, 1)  # 再次限制在 [0, 1] 范围内
        # inputs=inputs/0.5 -1.0
        # target[target< 0.0] = 0.0
        # target[target > 2.0] = 0.0
        return {'input': inputs, 'output': target}
class RandomCrop(object):
    """ Randomly crop 3-channel images
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        inputs, target = sample['input'], sample['output']
        b,c, h, w = inputs.shape  # 3-channel input: (C, H, W)
        th, tw = self.size

        # Ensure crop size doesn't exceed the input size
        if w < tw: 
            tw = w
        if h < th: 
            th = h

        # Randomly select the top-left corner for the crop
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        # Perform the crop
        inputs = inputs[:,:, y1:y1 + th, x1:x1 + tw]  # Apply crop across all channels
        target = target[:,y1:y1 + th, x1:x1 + tw]

        return {'input': inputs, 'output': target}


class RandomFilp(object):
    """ Randomly flip 3-channel images (Torch Tensors)
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, sample):
        inputs, target = sample['input'], sample['output']
        b, c, h, w = inputs.shape  # 3-channel input: (B, C, H, W)

        # Horizontal flip
        if torch.rand(1).item() < self.ratio:
            inputs = torch.flip(inputs, dims=[-1])  # Flip along the width (W)
            target = torch.flip(target, dims=[-1])  # Flip target along the width

        # Vertical flip
        if torch.rand(1).item() < self.ratio:
            inputs = torch.flip(inputs, dims=[-2])  # Flip along the height (H)
            target = torch.flip(target, dims=[-2])  # Flip target along the height

        return {'input': inputs.contiguous(), 'output': target.contiguous()}

class ResizeTransform:
    def __init__(self, size):
        self.resize = transforms.Resize(size)

    def __call__(self, sample):
        image, dpt = sample['input'], sample['output']

        # Apply resizing
        image = self.resize(image)
        dpt = self.resize(dpt)

        # Return as a dictionary
        return {'input': image, 'output': dpt}
class NormalizeTransform:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, dpt = sample['input'], sample['output']

        # 将图像和深度图归一化到 [-1, 1]
        image = image / 127.5 - 1.0

        # 返回处理后的字典
        return {'input': image, 'output': dpt}

def FoD500Loader(data_dir, n_stack=5, scale=1, focus_dist=[0.1,.15,.3,0.7,1.5]):

    img_train_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f[-7:] == "All.tif" and int(f[:6]) < 400]
    dpth_train_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f[-7:] == "Dpt.exr" and int(f[:6]) < 400]

    img_train_list.sort()
    dpth_train_list.sort()

    img_val_list =  [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f[-7:] == "All.tif" and int(f[:6]) >= 400]
    dpth_val_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f[-7:] == "Dpt.exr" and int(f[:6]) >=400]

    img_val_list.sort()
    dpth_val_list.sort()

    train_transform = transforms.Compose([
                        ToTensor(),
                        # ResizeTransform((224,224)),
                        RandomCrop(224),
                        RandomFilp(0.5),
                        RandomRotate(degrees=[0, 3]),  # Random rotation between -10 to 10 degrees
                        RandomColorJitter(contrast_range=(0.4,1.6), brightness_range=(-0.1, 0.1), gamma_range=(0.5,2.0))
                        ])
    dataset_train = ImageDataset(root_dir=data_dir, img_list=img_train_list, dpth_list=dpth_train_list,
                                 transform_fnc=train_transform, img_num=n_stack,  focus_dist=focus_dist, scale=scale)

    val_transform = transforms.Compose([ToTensor(),ResizeTransform((224,224))])
    dataset_valid = ImageDataset(root_dir=data_dir, img_list=img_val_list, dpth_list=dpth_val_list,
                                 transform_fnc=val_transform, img_num=n_stack, focus_dist=focus_dist, scale=scale)


    return dataset_train, dataset_valid
