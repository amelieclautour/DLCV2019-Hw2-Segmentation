import os
import json
import torch
import scipy.misc
import numpy


import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def is_image(filename):
    return filename.endswith('.png')


def load_image(file):
    return Image.open(file)



class DATA(Dataset):
    def __init__(self, args, mode='train'):

        """ set up basic parameters for dataset """
        self.mode = mode
        self.data_dir = args.data_dir
        if self.mode == 'train':
            self.img_dir = os.path.join(self.data_dir, 'train/img')
            self.seg_dir = os.path.join(self.data_dir, 'train/seg')
        elif self.mode == 'val':
            self.img_dir = os.path.join(self.data_dir, 'val/img')
            self.seg_dir = os.path.join(self.data_dir, 'val/seg')
        elif self.mode == 'test':
            self.img_dir = os.path.join(self.data_dir ,'..', 'img')
            self.seg_dir = os.path.join(self.data_dir,'..', 'seg')
        elif self.mode == 'save':
            self.img_dir = os.path.join(self.data_dir)

        ''' set up list of filenames for retrieval purposes (i.e 4-digit index: same for both img and seg files)'''
        self.filenames = [image_basename(f) for f in os.listdir(self.img_dir) if is_image(f)]
        self.filenames.sort()

        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                transforms.Normalize(MEAN, STD)
            ])

        elif self.mode == 'val' or self.mode == 'test' or self.mode == 'save':
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                transforms.Normalize(MEAN, STD)
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        """ get data """
        filename = self.filenames[idx]

        with open(os.path.join(self.img_dir, filename + '.png'), 'rb') as f:
            img = load_image(f).convert('RGB')
        if self.mode != 'save':
            with open(os.path.join(self.seg_dir, filename + '.png'), 'rb') as f:
                seg = load_image(f).convert('P')



        ''' transform image '''
        if self.transform is not None:
            img = self.transform(img)
            if self.mode != 'save' :
                seg = numpy.array(seg)
                seg = torch.from_numpy(seg)
                seg = seg.long()
        if self.mode == 'save'  :
            return img

        return img, seg