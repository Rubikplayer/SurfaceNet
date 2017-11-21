import numpy as np
import os
from scipy import misc
import sys

home_folder = '/home/mengqi/dataset/MVS'

def load_imgs_MVS(modelIndx):
    #preload all the rectified imgs of the model: modelIndx
    rectified_img_folder = os.path.join(home_folder, 'Rectified/scan'+str(modelIndx)+'/')
    rectified_img_mean_folder = os.path.join(home_folder, 'Rectified_mean/scan'+str(modelIndx)+'/')
    file_list = os.listdir(rectified_img_folder)
    if len(file_list)%8 != 0 or len(file_list) == 0:
        print('for each view there SHOULD be EIGHT imgs on different light conditions')
        return 0
    if not os.path.exists(rectified_img_mean_folder):
        os.makedirs(rectified_img_mean_folder)
    file_list.sort() # can be read in order
    for _viewIndx in range(1, len(file_list)/8 + 1): # forgot to +1 in the commit '341745cb' !
        for _lights in range(8):
            file = file_list[_viewIndx*8+_lights]
            if _lights == 0:
                imgtmp = misc.imread(rectified_img_folder+ file)
                img_shape = imgtmp.shape
                rectified_img_array = np.zeros((1, 8) + img_shape)
            img_color = misc.imread(rectified_img_folder+ file)
            rectified_img_array[0,_lights] = img_color

        rectified_img_mean_array = np.mean(rectified_img_array,axis=1)
        misc.imsave(rectified_img_mean_folder+'rect_{0:03}_mean.jpg'.format(_viewIndx),rectified_img_mean_array[0])
    print('rectified imgs are loaded')
    
try:
    load_imgs_MVS(int(sys.argv[1]))
except Exception: 
    print 'exception', sys.exc_info()
    pass
