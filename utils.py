import os
import numpy as np
import math
import cv2
from config import Configs

cfg = Configs().parse() 
SPLITSIZE  = cfg.split_size

def imvisualize(imdeg, imgt, impred, ind, epoch='0',setting=''):
    """
    Visualize the predicted images along with the degraded and clean gt ones

    Args:
        imdeg (tensor): degraded image
        imgt (tensor): gt clean image
        impred (tensor): prediced cleaned image
        ind (str): index of images (name)
        epoch (str): current epoch
        setting (str): experiment name
    """
    # unnormalize data
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

    # avoid taking values of pixels outside [0, 1]
    impred[np.where(impred>1)] = 1
    impred[np.where(impred<0)] = 0

    # create vis folder
    if not os.path.exists('vis'+setting+'/epoch'+epoch):
        os.makedirs('vis'+setting+'/epoch'+epoch)
    
    # binarize the predicted image taking 0.5 as threshold
    impred = (impred>0.5)*1

    # save images
    cv2.imwrite('vis'+setting+'/epoch'+epoch+'/'+ind.split('.')[0]+'_deg.png',imdeg*255)
    cv2.imwrite('vis'+setting+'/epoch'+epoch+'/'+ind.split('.')[0]+'_gt.png',imgt*255)
    cv2.imwrite('vis'+setting+'/epoch'+epoch+'/'+ind.split('.')[0]+'_pred.png',impred*255)
    
def psnr(img1, img2):
    """
    Count PSNR of two images

    Args:
        img1 (np.array): first image
        img2 (np.array): second image
    Returns:
        p (int): the PSNR value 
    """
    mse = np.mean( (img1 - img2) ** 2 )
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    p = (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
    
    return p

def reconstruct(idx, h, w, epoch, setting, flipped=False):
    """
    reconstruct DIBCO (or other) full images from the binarized patches

    Args:
        idx (str): name of the image
        h (int): height of original image to be constructed from patches
        w (int): width of original image to be constructed from patches
        epoch (int): current epoch
        setting (str): experiment name
        flipped (bool): if the images are flipped, reconstruct and flip
    Returns:
        rec_image (np.array): the reconstruted image 

    """
    # initialize image
    rec_image = np.zeros(((h//SPLITSIZE + 1)*SPLITSIZE,(w//SPLITSIZE + 1)*SPLITSIZE,3))
    
    # fill the image 
    for i in range (0,h,SPLITSIZE):
        for j in range(0,w,SPLITSIZE):
            p = cv2.imread('vis'+setting+'/epoch'+str(epoch)+'/'+idx+'_'+str(i)+'_'+str(j)+'_pred.png')
            if flipped:
                p = cv2.rotate(p, cv2.ROTATE_180)
            rec_image[i:i+SPLITSIZE,j:j+SPLITSIZE,:] = p
    
    # trim the image from padding
    rec_image =  rec_image[:h,:w,:]
    
    return rec_image

def count_psnr(epoch, data_path, valid_data='2018',setting='',flipped = False , thresh = 0.5):
    """
    reconstruct images and count the PSNR for the full validation dataset

    Args:
        epoch (int): current epoch
        data_path (str): path of the data folder
        valid_data (str): which validation dataset
        setting (str): experiment name
        flipped (bool): whether the images are flipped
        thresh (int): binarization threshold after cleaning
    Returns: 
        avg_psnr (float): the PSNR result of the full dataset image pairs
    """
    total_psnr = 0
    qo = 0
    
    gt_folder = data_path + 'DIBCOSETS/' + valid_data + '/gt_imgs' 
    gt_imgs = os.listdir(gt_folder)
    flip_status = 'flipped' if flipped else 'normal'
    
    if not os.path.exists('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status):
        os.makedirs('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status)

    for im in gt_imgs:
        gt_image = cv2.imread(gt_folder+'/'+im)
        max_p =  np.max(gt_image) # max_p is 1 or 255
        gt_image = gt_image / max_p
        pred_image = reconstruct(im.split('.')[0],gt_image.shape[0],gt_image.shape[1],epoch,setting, flipped = flipped)/ max_p
        pred_image = (pred_image>thresh)*1
        total_psnr+=psnr(pred_image,gt_image)
        qo+=1

        # save reconstructed cleaned image with the gt one.
        cv2.imwrite('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status+'/'+im,gt_image*255)
        cv2.imwrite('vis'+setting+'/epoch'+str(epoch)+'/00_reconstr_'+flip_status+'/'+im.split('.')[0]+'_pred.png',pred_image*255)

    avg_psnr = total_psnr/qo
    
    return avg_psnr