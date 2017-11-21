#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time
import math
import numpy as np


import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from PIL import Image
from scipy import misc

import prepare_data
import random
# import cPickle as pickle

random.seed(201512)


    
def iterate_minibatches(modelIndx, NOofViews, onlyDisplay):

    save_dataset_folder = '/home/mengqi/dataset/MVS/lasagne/save_inputs_target_{}views_triplet/'.format(NOofViews)
    save_dataset_angle_folder = '/home/mengqi/dataset/MVS/lasagne/save_inputs_target_{}views_triplet_angle/'.format(NOofViews)
    save_dataset_COSangle_folder = '/home/mengqi/dataset/MVS/lasagne/save_inputs_target_{}views_triplet_COSangle/'.format(NOofViews)
    rectified_img_folder = '/home/mengqi/dataset/MVS/Rectified_l3/scan'+str(modelIndx)+'/'
    sample_TXT_folder = '/home/mengqi/dataset/MVS/samplesTXT/'
    cameraT_folder = '/home/mengqi/dataset/MVS/cameraT/'
    cameraPos_folder = '/home/mengqi/dataset/MVS/pos/'

    # read all the cameraT in order to calculate the angle view1_3Dpoint_view2
    cameraT_array = prepare_data.load_cameraT_as_np(cameraT_folder)

    # read all the camera pos in order to calculate the negative anchor of the triplet. 
    # the negative patch pair should come from the same 3D point in free space, so does the positive pair.
    # and the neg/pos patch pair should share one patch in common.
    cameraPos_array = prepare_data.load_cameraPos_as_np(cameraPos_folder)

    if not os.path.exists(save_dataset_folder):
        os.makedirs(save_dataset_folder)
    if not os.path.exists(save_dataset_angle_folder):
        os.makedirs(save_dataset_angle_folder)
    if not os.path.exists(save_dataset_COSangle_folder):
        os.makedirs(save_dataset_COSangle_folder)
    #preload all the rectified imgs of the model: modelIndx
    file_list = os.listdir(rectified_img_folder)
    NO_views = len(file_list)


    rectified_img_list = [[None for _i1 in range(1)] for _i2 in range(len(file_list))]
    for file in file_list:
        viewIndx = int(file.split('.')[0].split('_')[1])

        rectified_img_list[viewIndx-1][0] = misc.imread(rectified_img_folder+ file)
    print('rectified imgs are loaded')

    for mode in ['hard']:
        # read the file containing the on/off surface samples' information
        with open(sample_TXT_folder+'output_stl_'+str(modelIndx).zfill(3)+'_'+mode+'.txt', 'rU') as file_train:
            samplefile = file_train.read().split('\n')[:-2] #ignore the last empty elements
            random.shuffle(samplefile)
            #samplefile = samplefile[:1000]
        # filter out the off surf sample points
        samplefile_on_surf = [_on_surf_pt for _on_surf_pt in samplefile if _on_surf_pt.split(':')[0]=='1']    # this step already filter out the off surface points
        #create a container to save the patches, need to determine the shape at the very beginning
        iteration_4_each_3Dpt = 2
        NO_of_3Dpts = len(samplefile_on_surf)
        inputs = np.zeros((NO_of_3Dpts * iteration_4_each_3Dpt,3,3,64,64),dtype=np.uint8) # inputs[i][j]: the ith set jth view
        triplet_angle = np.zeros((NO_of_3Dpts * iteration_4_each_3Dpt,2),dtype=np.float32) # inputs[i][j]: the ith set jth view
        triplet_COSangle = np.zeros((NO_of_3Dpts * iteration_4_each_3Dpt,2),dtype=np.float32) # inputs[i][j]: the ith set jth view
        inputs_counter = 0
        for _i_3D_pt in range(iteration_4_each_3Dpt):
            random.shuffle(samplefile_on_surf) # because there is no target any more, the shuffle operation is safe.
            for i, line in enumerate(samplefile_on_surf):
                #onsurf = lpcd_resolution = 1ine.split(':')[0]  
    
                xyz_str = line.split(':')[1].split(';')[-1].split(' ')
                xyz_3D_pt = np.asarray(xyz_str).astype(np.float32) # the xyz coordinates of the 3D point
                # read all 64 views' uv position into view_all, because the view_indx is start from 1, view_all[0]=[]
                view_all = [[]] + line.split(':')[1].split(';')[:-1]

                pt_is_valid = False 
                for _i_vis_check in range(20): # make sure the 3D point is not occluded in the 3 views.
                    # randomly select NOofviews ordered elements in the view list
                    indx_view1 = random.choice(range(1,NO_views+1)) 
                    indx_view2 = random.choice(range(1,NO_views+1)) 
                    indx_view3 = random.choice(range(1,NO_views+1)) 
                    visib_Left = view_all[indx_view1].split(' ')[-1]
                    visib_Right = view_all[indx_view2].split(' ')[-1]
                    visib_3rd = view_all[indx_view3].split(' ')[-1]
                    if visib_Left=='1' and visib_Right=='1' and visib_3rd=='1' and \
                            indx_view1!=indx_view2 and indx_view1!=indx_view3:  # because the 1&2 is the positive pair, 1&3 is the negative pair
                        pt_is_valid = True
                        break
                if not pt_is_valid : # this point is rarely visible.
                    continue
                
                    
                indx_view = [indx_view1, indx_view2]
                # use the indx_NOofviews as the indices of the list
                # view only stores the views which are really used for each sample (say 4-views)
                view = [view_all[indx_i] for indx_i in indx_view]
                
                # calculate angle view1-3Dpt-view2, (assume the there are 2 views in each set)
                vect_v1_pt = cameraT_array[indx_view1]-xyz_3D_pt
                angle_v1_pt_v2, cos_angle_v1_pt_v2 = prepare_data.calculate_angle_p1_p2_p3(p1=cameraT_array[indx_view1],\
                        p2=xyz_3D_pt, p3=cameraT_array[indx_view2])

                # to get an off-surf point sharing the same img patch in view1 (1st patch in the triplet)
                img_h, img_w = rectified_img_list[0][0].shape[:2]
                while True: # do-while in python
                    vect_diff = vect_v1_pt * random.uniform(-.01,.01) # add a vaberation to the on-surf point to get a pt in free space
                    if np.linalg.norm(vect_diff) > pcd_resolution:
                        xyz_3D_pt_free = xyz_3D_pt + vect_diff
                        neg_proj_w, neg_proj_h = prepare_data.perspectiveProj(cameraPos_array[indx_view3], xyz_3D_pt_free) # the negative patch in the triplet is also sampled in the indx_view2 in order to generate hard samples
                        if neg_proj_w > 0 and neg_proj_w < img_w and neg_proj_h > 0 and neg_proj_h < img_h:
                            break
                ## center_w, center_h = perspectiveProj(cameraPos_array[indx_view1], xyz_3D_pt_free) # for test, check whether the w/h is equal to the following w/h
                # calculate the 
                angle_v1_freept_v3, cos_angle_v1_freept_v3 = prepare_data.calculate_angle_p1_p2_p3(p1=cameraT_array[indx_view1],\
                        p2=xyz_3D_pt_free, p3=cameraT_array[indx_view3])

                light_rand = 0
                if onlyDisplay:
                    fig, axarr = plt.subplots(1, 6) # for display of the triplet
                for _, view_i in enumerate(view):
                    view_i_data = view_i.split(' ') # viewIdx, x, y, d, visibility
                    viewIndx = int(view_i_data[0])
                    matlab_x = view_i_data[1]
                    matlab_y = view_i_data[2]
                    depth = view_i_data[3]
                    visibility = view_i_data[-1]
    
                    img=rectified_img_list[viewIndx-1][light_rand]
                    h = int(matlab_y)
                    w = int(matlab_x)
                    #patch_r = int(10000/float(depth)) # patch size is inverse proportional to the depth of the voxel
                    patch_r = inputs.shape[-1]/2
                    patch_r_Large = patch_r + patch_r/2 # to make sure the final patch got from rotation and crop has no black area
                    
                    # PIL.crop: 4-tuple defining the left, upper, right, and lower pixel coordinate, can access out of img range
                    patch_before_rotat = Image.fromarray(img).crop((w-patch_r_Large, h-patch_r_Large, w+patch_r_Large+1, h+patch_r_Large+1))
                    rand_angle = random.uniform(-10,10)
                    # rotation variance
                    patch_rotat = patch_before_rotat.rotate(rand_angle, resample=Image.BICUBIC)
                    # scale variance
                    patch_r_random = patch_r + random.randint(-1*patch_r/8, patch_r/8) # for random crop
                    patch_rand_crop = patch_rotat.crop((patch_r_Large - patch_r_random, patch_r_Large - patch_r_random,
                                                        patch_r_Large + patch_r_random+1, patch_r_Large + patch_r_random+1))
                    patch = patch_rand_crop.resize((inputs.shape[-2:]), Image.ANTIALIAS)
                    im = np.array(patch,dtype=np.uint8)
    
                    # preprocess the img for VGG
                    # Shuffle axes to c01: channel,height,weigth
                    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
                    # Convert to BGR
                    im = im[::-1, :, :]
    
                    inputs[inputs_counter, _] = im
                #subplot
                    if onlyDisplay:
                        axarr[_].imshow(patch)
                        axarr[_].axis('off')
                        axarr[_].set_title('cos1={:.3}, angle1={:.3}'.format(cos_angle_v1_pt_v2,angle_v1_pt_v2))

                # after the anchor and the positive sample are generated, try to get the negative sample
                # w_rand = random.randint(patch_r, patch_r_Large-patch_r/4) # random crop center
                # h_rand = random.randint(patch_r, patch_r_Large-patch_r/4)
                patch = Image.fromarray(img).crop((neg_proj_w-patch_r, neg_proj_h-patch_r, neg_proj_w+patch_r, neg_proj_h+patch_r))
                im = np.array(patch, dtype=np.uint8)
                im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
                im = im[::-1, :, :]   # Convert to BGR             
                inputs[inputs_counter, 2] = im
                triplet_angle[inputs_counter, 0] = angle_v1_pt_v2
                triplet_angle[inputs_counter, 1] = angle_v1_freept_v3
                triplet_COSangle[inputs_counter, 0] = cos_angle_v1_pt_v2
                triplet_COSangle[inputs_counter, 1] = cos_angle_v1_freept_v3
                inputs_counter += 1
                if onlyDisplay:
                    axarr[2].imshow(patch)
                    axarr[2].axis('off')
                    axarr[2].set_title("cos2={:.3}, angle2={:.3},v123={},{},{}".format(cos_angle_v1_freept_v3,angle_v1_freept_v3,indx_view1,indx_view2,indx_view3))                
                
                
                if onlyDisplay:
                    plt.show()
                if inputs_counter%6000 == 0:
                    print('{}: {}th set'.format(modelIndx, inputs_counter))

        if not onlyDisplay:
            np.save(open(save_dataset_folder+str(modelIndx).zfill(3)+'_'+mode+'_inputs.data', "wb" ), inputs[:inputs_counter])
            np.save(open(save_dataset_angle_folder+str(modelIndx).zfill(3)+'_'+mode+'_inputs.data', "wb" ), triplet_angle[:inputs_counter])
            np.save(open(save_dataset_COSangle_folder+str(modelIndx).zfill(3)+'_'+mode+'_inputs.data', "wb" ), triplet_COSangle[:inputs_counter])

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
visualizationON = False
pcd_resolution = 1 # the min distance between the on/off_surf points in each triplet 

if visualizationON:
    iterate_minibatches(3, NOofViews=2, onlyDisplay = visualizationON)
else:   
    try:
        iterate_minibatches(int(sys.argv[1]), NOofViews=2, onlyDisplay = visualizationON)
    except Exception: 
        print("exception"+str(sys.exc_info()))
        pass

