import torch
import torch.utils.data as D
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import os
from PIL import Image
from config import Configs

class Read_data(D.Dataset):
    """
    The data loader class for 1 set (for example train)

    Args:
        base_dir (str): the data path
        file_label (list of str): the names of the data instances
        set (str): the set (train, valid or test)
        split_size (int): the image (patch) size
        augmentation (bool): whwther to apply augmentation
        flipped (bool): whether the data is flipped
    """
    def __init__(self, base_dir, file_label,set, split_size, augmentation=True , flipped = False):
        self.base_dir = base_dir
        self.file_label = file_label
        self.set = set
        self.split_size = split_size
        self.augmentation = augmentation
        self.flipped  = flipped
        
    def __getitem__(self, index):
        img_name = self.file_label[index]
        idx, deg_img, gt_img = self.readImages(img_name)
        return idx, deg_img, gt_img
        
    def __len__(self):
        return len(self.file_label)

    def readImages(self, file_name):
        """
        Read a pair of images (degraded + clean gt)
        
        Args:
            file_name (str): the index (name) of the image pair
        Returns:
            file_name (str): the index (name) of the image pair
            out_deg_img (np.array): the degraded image
            out_gt_img (np.array): the clean image

        """
        url_deg = self.base_dir +'/'+ self.set+'/' + file_name
        url_gt = self.base_dir +'/'+ self.set+'_gt/'+file_name
        
        deg_img = cv2.imread(url_deg)
        gt_img = cv2.imread(url_gt)

        if self.flipped:
            deg_img = cv2.rotate(deg_img, cv2.ROTATE_180)
            gt_img = cv2.rotate(gt_img, cv2.ROTATE_180)
        try:
            deg_img.any()
        except:
            print('###!Cannot find image: ' + url_deg)
        try:
            gt_img.any()
        except:
            print('###!Cannot find image: ' + url_gt)
        
        deg_img = Image.fromarray(np.uint8(deg_img))
        gt_img = Image.fromarray(np.uint8(gt_img))

        # apply data augmentation
        if self.augmentation:
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(deg_img, output_size=(self.split_size, self.split_size))
            deg_img = TF.crop(deg_img, i, j, h, w)
            gt_img = TF.crop(gt_img, i, j, h, w)

            # random horizontal flipping
            if random.random() > 0.5:
                deg_img = TF.hflip(deg_img)
                gt_img = TF.hflip(gt_img)

            # random vertical flipping
            if random.random() > 0.5:
                deg_img = TF.vflip(deg_img)
                gt_img = TF.vflip(gt_img)

        deg_img = (np.array(deg_img) /255).astype('float32')
        gt_img = (np.array(gt_img)  / 255).astype('float32')
        
        # normalize data
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        out_deg_img = np.zeros([3, *deg_img.shape[:-1]])
        out_gt_img = np.zeros([3, *gt_img.shape[:-1]])
        for i in range(3):
            out_deg_img[i] = (deg_img[:,:,i] - mean[i]) / std[i]
            out_gt_img[i] = (gt_img[:,:,i] - mean[i]) / std[i]
        
        return file_name, out_deg_img, out_gt_img


def load_datasets(flipped=False):
    """
    Create the 3 datasets (train/valid/test) to be used by the dataloaders.

    Args:
        flipped (bool): whwther to flip the images of the val dataset (was used
                        in 1 experiment to check the effect of flipping)
    Returns:
        data_train (Dateset): train data
        data_valid (Dateset): valid data
        data_test (Dateset): test data
    """
    cfg = Configs().parse() 
    base_dir = cfg.data_path
    split_size  = cfg.split_size
    data_tr = os.listdir(cfg.data_path+'train')
    np.random.shuffle(data_tr)
    data_va = os.listdir(cfg.data_path+'valid')
    np.random.shuffle(data_va)
    data_te = os.listdir(cfg.data_path+'test')
    np.random.shuffle(data_te)
    
    data_train = Read_data(base_dir, data_tr, 'train', split_size, augmentation=True)
    data_valid = Read_data(base_dir, data_va, 'valid', split_size, augmentation=False, flipped = flipped)
    data_test = Read_data(base_dir, data_te, 'test', split_size, augmentation=False)

    return data_train, data_valid, data_test

def sort_batch(batch):
    """
    Transform a batch of data to pytorch tensor

    Args:
        batch [str, np.array, np.array]: a batch of data
    Returns:
        data_index (tensor): the indexes of the source/target pair
        data_in (tensor): the source images (degraded)
        data_out (tensor): the target images (clean gt)
    """
    n_batch = len(batch)
    data_index = []
    data_in = []
    data_out = []
    for i in range(n_batch):
        idx, img, gt_img = batch[i]

        data_index.append(idx)
        data_in.append(img)
        data_out.append(gt_img)

    data_index = np.array(data_index)
    data_in = np.array(data_in, dtype='float32')
    data_out = np.array(data_out, dtype='float32')

    data_in = torch.from_numpy(data_in)
    data_out = torch.from_numpy(data_out)

    return data_index, data_in, data_out

def all_data_loader(batch_size):
    """
    Create the 3 data loaders

    Args:
        batch_size (int): the batch_size
    Returns:
        train_loader (dataloader): train data loader  
        valid_loader (dataloader): valid data loader
        test_loader (dataloader): test data loader
    """
    data_train, data_valid, data_test = load_datasets()
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, valid_loader, test_loader