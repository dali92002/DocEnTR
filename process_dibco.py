import numpy as np
import os
import random
import cv2
from tqdm import tqdm
from config  import Configs

def prepare_dibco_experiment(val_set,test_set, patches_size, overlap_size, patches_size_valid):
    """
    Prepare the data for training

    Args:
        val_set (str): the vealidation dataset
        the_set (str): the testing dataset
        patches_size (int): patch size for training data
        overlap_size (int): overlapping size between different patches (vertically and horizontally)
        patches_size_valid (int): patch size for validation data
    """
    folder = main_path+'DIBCOSETS/'
    all_datasets = os.listdir(folder)
    n_i = 1

    for d_set in tqdm(all_datasets):
        if d_set not in  [val_set,test_set]:
            # continue
            for im in os.listdir(folder+d_set+'/imgs'):
                img = cv2.imread(folder+d_set+'/imgs/'+im)
                gt_img = cv2.imread(folder+d_set+'/gt_imgs/'+im)

                for i in range (0,img.shape[0],overlap_size):
                    for j in range (0,img.shape[1],overlap_size):

                        if i+patches_size<=img.shape[0] and j+patches_size<=img.shape[1]:
                            p = img[i:i+patches_size,j:j+patches_size,:]
                            gt_p = gt_img[i:i+patches_size,j:j+patches_size,:]
                        
                        elif i+patches_size>img.shape[0] and j+patches_size<=img.shape[1]:
                            p = (np.ones((patches_size,patches_size,3)) - random.randint(0,1) )*255
                            gt_p = np.ones((patches_size,patches_size,3)) *255
                            
                            p[0:img.shape[0]-i,:,:] = img[i:img.shape[0],j:j+patches_size,:]
                            gt_p[0:img.shape[0]-i,:,:] = gt_img[i:img.shape[0],j:j+patches_size,:]
                        
                        elif i+patches_size<=img.shape[0] and j+patches_size>img.shape[1]:
                            p = (np.ones((patches_size,patches_size,3)) - random.randint(0,1) )*255
                            gt_p = np.ones((patches_size,patches_size,3)) * 255
                            
                            p[:,0:img.shape[1]-j,:] = img[i:i+patches_size,j:img.shape[1],:]
                            gt_p[:,0:img.shape[1]-j,:] = gt_img[i:i+patches_size,j:img.shape[1],:]

                        else:
                            p = (np.ones((patches_size,patches_size,3)) - random.randint(0,1) )*255
                            gt_p = np.ones((patches_size,patches_size,3)) * 255
                            
                            p[0:img.shape[0]-i,0:img.shape[1]-j,:] = img[i:img.shape[0],j:img.shape[1],:]
                            gt_p[0:img.shape[0]-i,0:img.shape[1]-j,:] = gt_img[i:img.shape[0],j:img.shape[1],:]


                        
                        cv2.imwrite(main_path+'train/'+str(n_i)+'.png',p)
                        cv2.imwrite(main_path+'train_gt/'+str(n_i)+'.png',gt_p)
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

            
                        cv2.imwrite(main_path+'test/'+im.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',p)
                        cv2.imwrite(main_path+'test_gt/'+im.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',gt_p)

        if d_set == val_set:
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

            
                        cv2.imwrite(main_path+'valid/'+im.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',p)
                        cv2.imwrite(main_path+'valid_gt/'+im.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',gt_p)


if __name__ == "__main__":
    # get configs 
    cfg = Configs().parse()
    main_path = cfg.data_path
    validation_dataset = cfg.validation_dataset
    testing_dataset = cfg.testing_dataset
    patch_size =  cfg.split_size
    # augment the training data patch size to allow cropping augmentation later in data loader
    p_size_train = (patch_size+128)
    p_size_valid  = patch_size
    overlap_size = patch_size//2

    # create train/val/test folders if theu are not existent
    if not os.path.exists(main_path+'train/'):
        os.makedirs(main_path+'train/')
    if not os.path.exists(main_path+'train_gt/'):
        os.makedirs(main_path+'train_gt/')

    if not os.path.exists(main_path+'valid/'):
        os.makedirs(main_path+'valid/')
    if not os.path.exists(main_path+'valid_gt/'):
        os.makedirs(main_path+'valid_gt/')

    if not os.path.exists(main_path+'test/'):
        os.makedirs(main_path+'test/')
    if not os.path.exists(main_path+'test_gt/'):
        os.makedirs(main_path+'test_gt/')

    # remove old data if the folders exist
    os.system('rm '+main_path+'train/*')
    os.system('rm '+main_path+'train_gt/*')
                    
    os.system('rm '+main_path+'valid/*')
    os.system('rm '+main_path+'valid_gt/*')

    os.system('rm '+main_path+'test/*')
    os.system('rm '+main_path+'test_gt/*')

    # create your data...
    prepare_dibco_experiment(validation_dataset, testing_dataset, p_size_train, overlap_size, p_size_valid)

