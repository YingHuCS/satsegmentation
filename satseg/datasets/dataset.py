import os
import torch
from PIL import Image
import numpy as np



class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform


    def __len__(self):
        return len(os.listdir(self.img_dir))


    def __getitem__(self, i):
        img_name = os.listdir(self.img_dir)[i]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)

        mask_suf = os.listdir(self.mask_dir)[0].split('.')[1] 
        mask_path = os.path.join(self.mask_dir, img_name.split('.')[0]+'.'+mask_suf)
        mask = Image.open(mask_path)
        
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        
        return img, mask



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform


    def __len__(self):
        return len(os.listdir(self.img_dir))


    def __getitem__(self, i):
        img_name = os.listdir(self.img_dir)[i]        
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)


        if self.transform is not None:
            img = self.transform(img)
        return img, img_name

