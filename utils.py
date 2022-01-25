import matplotlib.pyplot as plt
import os

import numpy as np
import math
from tqdm import tqdm
import cv2
from config import Configs

cfg = Configs().parse() 

SPLITSIZE  = cfg.split_size



def imvisualize(imdeg,imgt,impred,ind,epoch='0',setting=''):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    imdeg = imdeg.numpy()
    imgt = imgt.numpy()
    impred = impred.numpy()

    imdeg = np.transpose(imdeg, (1, 2, 0))
    imgt = np.transpose(imgt, (1, 2, 0))
    impred = np.transpose(impred, (1, 2, 0))
    
    for ch in range(3):
        imdeg[:,:,ch] = (imdeg[:,:,ch] *std[ch]) + mean[ch]
        imgt[:,:,ch] = (imgt[:,:,ch] *std[ch]) + mean[ch]
        impred[:,:,ch] = (impred[:,:,ch] *std[ch]) + mean[ch]

    impred[np.where(impred>1)] = 1
    impred[np.where(impred<0)] = 0

    if not os.path.exists('vis'+setting+'/epoch'+epoch):
        os.makedirs('vis'+setting+'/epoch'+epoch)
    
    ### binarize

    impred = (impred>0.5)*1

    
    cv2.imwrite('vis'+setting+'/epoch'+epoch+'/'+ind.split('.')[0]+'_deg.png',imdeg*255)
    cv2.imwrite('vis'+setting+'/epoch'+epoch+'/'+ind.split('.')[0]+'_gt.png',imgt*255)
    cv2.imwrite('vis'+setting+'/epoch'+epoch+'/'+ind.split('.')[0]+'_pred.png',impred*255)
    

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))




def reconstruct(idx,h,w,epoch,setting,flipped=False):

    rec_image = np.zeros(((h//SPLITSIZE + 1)*SPLITSIZE,(w//SPLITSIZE + 1)*SPLITSIZE,3))

    for i in range (0,h,SPLITSIZE):
        for j in range(0,w,SPLITSIZE):
            p = cv2.imread('vis'+setting+'/epoch'+str(epoch)+'/'+idx+'_'+str(i)+'_'+str(j)+'_pred.png')
            if flipped:
                p = cv2.rotate(p, cv2.ROTATE_180)
            rec_image[i:i+SPLITSIZE,j:j+SPLITSIZE,:] = p

    rec_image =  rec_image[:h,:w,:]
    return rec_image




def count_psnr(epoch, valid_data='2018',setting='',flipped = False , thresh = 0.5):
    avg_psnr = 0
    qo = 0
    folder = 'vis/epoch'+str(epoch)
    gt_folder = 'data/DIBCOSETS/'+valid_data+'/gt_imgs' 

    gt_imgs = os.listdir(gt_folder)

    flip_status = 'flipped' if flipped else 'normal'
    
    # pred_imgs = os.listdir(folder)
    # pred_imgs = [im for im  in pred_imgs if 'pred' in im]

    if not os.path.exists('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status):
        os.makedirs('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status)


    for im in gt_imgs:


        gt_image = cv2.imread(gt_folder+'/'+im)
        max_p =  np.max(gt_image)
        gt_image = gt_image / max_p
        
        

        pred_image = reconstruct(im.split('.')[0],gt_image.shape[0],gt_image.shape[1],epoch,setting, flipped = flipped)/ max_p
        # pred_image = cv2.rotate(pred_image, cv2.ROTATE_180)

        pred_image = (pred_image>thresh)*1
        avg_psnr+=psnr(pred_image,gt_image)
        qo+=1

        cv2.imwrite('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status+'/'+im,gt_image*255)
        cv2.imwrite('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status+'/'+im.split('.')[0]+'_pred.png',pred_image*255)
        

    return(avg_psnr/qo)        


def count_psnr_both(epoch, valid_data='2018',setting='',thresh = 0.5):
    avg_psnr = 0
    qo = 0
    folder = 'vis/epoch'+str(epoch)
    gt_folder = 'data/DIBCOSETS/'+valid_data+'/gt_imgs' 

    gt_imgs = os.listdir(gt_folder)

    
    if not os.path.exists('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_merge'):
        os.makedirs('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_merge')


    for im in gt_imgs:


        gt_image = cv2.imread(gt_folder+'/'+im)
        max_p =  np.max(gt_image)
        gt_image = gt_image / max_p
        
        

        pred_image1 = cv2.imread('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_flipped/'+im.replace('.png','_pred.png'))/ max_p
        pred_image2 = cv2.imread('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_normal/'+im.replace('.png','_pred.png'))/ max_p

        pred_image1 = (pred_image1>thresh)*1
        pred_image2 = (pred_image2>thresh)*1


        pred_image = np.ones((gt_image.shape[0],gt_image.shape[1],gt_image.shape[2]))

        for i in range(gt_image.shape[0]):
            for j in range(gt_image.shape[1]):
                for k in range(gt_image.shape[2]):
                    if pred_image1[i,j,k] == 1  or pred_image2[i,j,k] == 1: 
                        pred_image[i,j,k] = 1
                    else:
                        pred_image[i,j,k] = pred_image1[i,j,k] + pred_image2[i,j,k]
                    

        
        avg_psnr+=psnr(pred_image,gt_image)
        qo+=1

        cv2.imwrite('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_merge'+'/'+im,gt_image*255)
        cv2.imwrite('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_merge'+'/'+im.split('.')[0]+'_pred.png',pred_image*255)
        

    return(avg_psnr/qo)        


# print(count_psnr_both(0,setting='base_256_8'))

# a=444
# import csv

# import pandas
# df = pandas.DataFrame(data={"epoch": ep_nums, 'psnr': psnr_list})
# df.to_csv("./psnr.csv", sep=',',index=False)