# The location should be <your_location_to_the_dataset>, so you need to 
# make sure that all the groundtruth files can be found in that location, 
# a folder "words/" should also be there, which contains all the images
# without any folder or sub-folder.

import cv2

baseDir_binarize = '/home/mohamed/vit/binarization/data'


# import os


# images = os.listdir(baseDir_binarize+'/valid_gt/')

# for im in images:
#     img = cv2.imread(baseDir_binarize+'/valid_gt/'+im)
#     img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
#     cv2.imwrite(baseDir_binarize+'/valid_gt/'+im,img)
    

# exit(0)