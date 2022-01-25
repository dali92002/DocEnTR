import torch.utils.data as D
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import os
from PIL import Image
from config import Configs

cfg = Configs().parse() 

split_size  = cfg.split_size

baseDir = cfg.data_path




class Read_data(D.Dataset):
    def __init__(self, file_label,set, augmentation=True , flipped = False):
        self.file_label = file_label
        self.set = set
        self.augmentation = augmentation
        self.flipped  = flipped
        
    def __getitem__(self, index):
        img_name = self.file_label[index]
        
        idx, deg_img, gt_img = self.readImages(img_name)
        return idx, deg_img, gt_img
        
    def __len__(self):
        return len(self.file_label)

    def readImages(self, file_name):
        file_name = file_name
        url_deg = baseDir +'/'+ self.set+'/' + file_name
        url_gt = baseDir +'/'+ self.set+'_gt/'+file_name
        
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

        if self.augmentation: # augmentation for training data
            
            # # Resize -->  it will affect the performance :3
            # resize = transforms.Resize(size=(256+64, 256+64))

            # deg_img = resize(deg_img)
            # gt_img = resize(gt_img)


            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(deg_img, output_size=(split_size, split_size))

            deg_img = TF.crop(deg_img, i, j, h, w)
            gt_img = TF.crop(gt_img, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                deg_img = TF.hflip(deg_img)
                gt_img = TF.hflip(gt_img)

            # Random vertical flipping
            if random.random() > 0.5:
                deg_img = TF.vflip(deg_img)
                gt_img = TF.vflip(gt_img)

        deg_img = (np.array(deg_img) /255).astype('float32')
        gt_img = (np.array(gt_img)  / 255).astype('float32')
        
        # if VGG_NORMAL:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        out_deg_img = np.zeros([3, *deg_img.shape[:-1]])
        out_gt_img = np.zeros([3, *gt_img.shape[:-1]])
        
        for i in range(3):
            out_deg_img[i] = (deg_img[:,:,i] - mean[i]) / std[i]
            out_gt_img[i] = (gt_img[:,:,i] - mean[i]) / std[i]
            
        
        return file_name, out_deg_img, out_gt_img

def loadData_sets(flipped=False):
    
    data_tr = os.listdir('data/train')
    np.random.shuffle(data_tr)
    data_va = os.listdir('data/valid')
    np.random.shuffle(data_va)
    data_te = os.listdir('data/test')
    np.random.shuffle(data_te)
    
    data_train = Read_data(data_tr,'train', augmentation=True)
    data_valid = Read_data(data_va,'valid',augmentation=False,flipped = flipped)
    data_test = Read_data(data_te,'test',augmentation=False)

    return data_train, data_valid, data_test