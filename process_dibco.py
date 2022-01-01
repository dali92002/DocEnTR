import numpy as np
import os 
from  PIL import Image
import random
import cv2
from shutil import copy
from tqdm import tqdm

# year = '2018'
# folder = 'data/DIBCO/'+year

# if not os.path.exists(folder+'/imgs'):
#     os.makedirs(folder+'/imgs')
#     os.makedirs(folder+'/gt_imgs')


# im_f = 'dibco2018_Dataset'
# img_gt_f = 'gt'


# imgs = os.listdir(folder+'/'+im_f)
# gt_imgs = os.listdir(folder+'/'+img_gt_f)


# for im in imgs:
#     img = Image.open(folder+'/'+im_f+'/'+im).convert('L')
#     img.save(folder+'/imgs/'+im.split('.')[0]+'.png')


# for gt_im in gt_imgs:
#     gt_img = Image.open(folder+'/'+img_gt_f+'/'+gt_im).convert('L')
#     gt_img.save(folder+'/gt_imgs/'+gt_im.split('.')[0]+'.png')


# folder = 'data/DIBCO/'
# years = os.listdir(folder)

# for year in years:
#     os.makedirs('DATA/DIBCOSETS/'+year)
#     os.makedirs('DATA/DIBCOSETS/'+year+'/imgs')
#     os.makedirs('DATA/DIBCOSETS/'+year+'/gt_imgs')
#     i=1
    
#     im_list = os.listdir(folder+year+'/imgs')
#     im_list.sort()
#     for im in im_list:
#         copy(folder+year+'/imgs/'+im,'DATA/DIBCOSETS/'+year+'/imgs/'+str(i)+'.png')
#         i+=1
#     i=1
#     gt_list = os.listdir(folder+year+'/gt_imgs')
#     gt_list.sort()
#     for im in gt_list:
#         copy(folder+year+'/gt_imgs/'+im,'DATA/DIBCOSETS/'+year+'/gt_imgs/'+str(i)+'.png')
#         i+=1




def prepare_dibco_experiment(test_set, patches_size, overlap_size, patches_size_valid):
    
    folder = 'data/DIBCOSETS/'

    all_datasets = os.listdir(folder)
    n_i = 1

    for d_set in tqdm(all_datasets):
        if d_set != test_set:
            for im in os.listdir(folder+d_set+'/imgs'):
                img = cv2.imread(folder+d_set+'/imgs/'+im)
                gt_img = cv2.imread(folder+d_set+'/gt_imgs/'+im)

                for i in range (0,img.shape[0],overlap_size):
                    for j in range (0,img.shape[1],overlap_size):

                        if i+patches_size<=img.shape[0] and j+patches_size<=img.shape[1]:
                            p = img[i:i+patches_size,j:j+patches_size,:]
                            gt_p = gt_img[i:i+patches_size,j:j+patches_size,:]
                        
                        elif i+patches_size>img.shape[0] and j+patches_size<=img.shape[1]:
                            p = np.ones((patches_size,patches_size,3)) *255
                            gt_p = np.ones((patches_size,patches_size,3)) *255
                            
                            p[0:img.shape[0]-i,:,:] = img[i:img.shape[0],j:j+patches_size,:]
                            gt_p[0:img.shape[0]-i,:,:] = gt_img[i:img.shape[0],j:j+patches_size,:]
                        
                        elif i+patches_size<=img.shape[0] and j+patches_size>img.shape[1]:
                            p = np.ones((patches_size,patches_size,3)) * 255
                            gt_p = np.ones((patches_size,patches_size,3)) * 255
                            
                            p[:,0:img.shape[1]-j,:] = img[i:i+patches_size,j:img.shape[1],:]
                            gt_p[:,0:img.shape[1]-j,:] = gt_img[i:i+patches_size,j:img.shape[1],:]

                        else:
                            p = np.ones((patches_size,patches_size,3)) * 255
                            gt_p = np.ones((patches_size,patches_size,3)) * 255
                            
                            p[0:img.shape[0]-i,0:img.shape[1]-j,:] = img[i:img.shape[0],j:img.shape[1],:]
                            gt_p[0:img.shape[0]-i,0:img.shape[1]-j,:] = gt_img[i:img.shape[0],j:img.shape[1],:]


                        
                        cv2.imwrite('data/train/'+str(n_i)+'.png',p)
                        cv2.imwrite('data/train_gt/'+str(n_i)+'.png',gt_p)
                        n_i+=1
        if d_set == test_set:
            for im in os.listdir(folder+d_set+'/imgs'):
                img = cv2.imread(folder+d_set+'/imgs/'+im)
                gt_img = cv2.imread(folder+d_set+'/gt_imgs/'+im)

                for i in range (0,img.shape[0],patches_size_valid):
                    for j in range (0,img.shape[1],patches_size_valid):

                        if i+patches_size_valid<=img.shape[0] and j+patches_size_valid<=img.shape[1]:
                            p = img[i:i+patches_size_valid,j:j+patches_size_valid,:]
                            gt_p = gt_img[i:i+patches_size_valid,j:j+patches_size_valid,:]
                        
                        elif i+patches_size_valid>img.shape[0] and j+patches_size_valid<=img.shape[1]:
                            p = np.ones((patches_size_valid,patches_size_valid,3)) *255
                            gt_p = np.ones((patches_size_valid,patches_size_valid,3)) *255
                            
                            p[0:img.shape[0]-i,:,:] = img[i:img.shape[0],j:j+patches_size_valid,:]
                            gt_p[0:img.shape[0]-i,:,:] = gt_img[i:img.shape[0],j:j+patches_size_valid,:]
                        
                        elif i+patches_size_valid<=img.shape[0] and j+patches_size_valid>img.shape[1]:
                            p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                            gt_p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                            
                            p[:,0:img.shape[1]-j,:] = img[i:i+patches_size_valid,j:img.shape[1],:]
                            gt_p[:,0:img.shape[1]-j,:] = gt_img[i:i+patches_size_valid,j:img.shape[1],:]

                        else:
                            p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                            gt_p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                            
                            p[0:img.shape[0]-i,0:img.shape[1]-j,:] = img[i:img.shape[0],j:img.shape[1],:]
                            gt_p[0:img.shape[0]-i,0:img.shape[1]-j,:] = gt_img[i:img.shape[0],j:img.shape[1],:]

            
                        cv2.imwrite('data/valid/'+im.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',p)
                        cv2.imwrite('data/valid_gt/'+im.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',gt_p)



            a=414




os.system('rm data/train/*')
os.system('rm data/train_gt/*')
                
os.system('rm data/valid/*')
os.system('rm data/valid_gt/*')


p_size = (512+128)
p_size_valid  = 512
overlap_size = 256

prepare_dibco_experiment('2018',p_size,overlap_size,p_size_valid)



exit(0)
