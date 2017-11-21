from shutil import copyfile
import numpy as np
import os
from scipy import misc
import sys


home_folder = '/home/mengqi/dataset/MVS'
light = 3
def selectLightIMG(modelIndx):
    input_folder = os.path.join(home_folder, 'Rectified/scan'+str(modelIndx))
    output_folder = os.path.join(home_folder, 'Rectified_l{}/scan{}'.format(light, modelIndx))
    if os.path.exists(input_folder) and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for viewIndx in range(80):
        input_img = os.path.join(input_folder, 'rect_{:03}_{}_r5000.png'.format(viewIndx, light))
        if os.path.isfile(input_img):
            output_img = os.path.join(output_folder, 'rect_{0:03}.png'.format(viewIndx))
            copyfile(input_img, output_img)
            # img = misc.imread(input_img)
            # misc.imsave(os.path.join(output_folder, 'rect_{0:03}.png'.format(viewIndx)), img)
            print('saved image file: ' + output_img)

try:
    selectLightIMG(modelIndx= int(sys.argv[1]))
except Exception: 
    print("exception"+str(sys.exc_info()))
    pass

