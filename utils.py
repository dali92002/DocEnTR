import matplotlib.pyplot as plt
import os

import numpy as np
import math
from tqdm import tqdm
import cv2
import config as cfg

SPLITSIZE = cfg.SPLITSIZE



def imvisualize(imdeg,imgt,impred,ind,epoch='0'):
    
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

    if not os.path.exists('vis/epoch'+epoch):
        os.makedirs('vis/epoch'+epoch)
    
    
    plt.imsave('vis/epoch'+epoch+'/'+ind.split('.')[0]+'_deg.png',imdeg)
    plt.imsave('vis/epoch'+epoch+'/'+ind.split('.')[0]+'_gt.png',imgt)
    plt.imsave('vis/epoch'+epoch+'/'+ind.split('.')[0]+'_pred.png',impred)
    

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))


def reconstruct(idx,h,w,epoch):

    rec_image = np.zeros(((h//SPLITSIZE + 1)*SPLITSIZE,(w//SPLITSIZE + 1)*SPLITSIZE,3))

    for i in range (0,h,SPLITSIZE):
        for j in range(0,w,SPLITSIZE):
            p = cv2.imread('vis/epoch'+str(epoch)+'/'+idx+'_'+str(i)+'_'+str(j)+'_pred.png')
            
            rec_image[i:i+SPLITSIZE,j:j+SPLITSIZE,:] = p

    return rec_image[:h,:w,:]




def count_psnr(epoch, valid_data='2018'):
    avg_psnr = 0
    qo = 0
    folder = 'vis/epoch'+str(epoch)
    gt_folder = 'data/DIBCOSETS/'+valid_data+'/gt_imgs' 

    gt_imgs = os.listdir(gt_folder)
    
    # pred_imgs = os.listdir(folder)
    # pred_imgs = [im for im  in pred_imgs if 'pred' in im]

    if not os.path.exists('vis/epoch'+str(epoch)+'/00_reconstr'):
        os.makedirs('vis/epoch'+str(epoch)+'/00_reconstr')

    for im in gt_imgs:


        gt_image = cv2.imread(gt_folder+'/'+im)
        max_p =  np.max(gt_image)
        gt_image = gt_image / max_p
        
        

        pred_image = reconstruct(im.split('.')[0],gt_image.shape[0],gt_image.shape[1],epoch)/ max_p


        pred_image = (pred_image>0.5)*1
        avg_psnr+=psnr(pred_image,gt_image)
        qo+=1

        cv2.imwrite('vis/epoch'+str(epoch)+'/00_reconstr/'+im,gt_image*255)
        cv2.imwrite('vis/epoch'+str(epoch)+'/00_reconstr/'+im.split('.')[0]+'_pred.png',pred_image*255)
        

    return(avg_psnr/qo)        


# print(count_psnr(1))

# a=444
# import csv

# import pandas
# df = pandas.DataFrame(data={"epoch": ep_nums, 'psnr': psnr_list})
# df.to_csv("./psnr.csv", sep=',',index=False)