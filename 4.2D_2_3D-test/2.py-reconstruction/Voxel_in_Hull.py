import numpy as np
import os
from scipy import misc
import random
from PIL import Image
import struct
import time
import matplotlib
import itertools
import sys
sys.path.append("../../3.2D_2_3D-train/2.2D_2_3D-train")
import params_volume as param

model_imgs_fld = param.__model_imgs_fld
view_set = param.__view_set


class voxel_in_hull:
    def __init__(self, datasetName, modelIndx, MEAN_IMAGE_BGR, views_4_reconstr, patch_size = 64, xyz_np = None):
        from train_val_data_4Test import all_viewPair_viewIndx, all_viewPair_dimIndx_np
        self.datasetName = datasetName
        self.MEAN_IMAGE_BGR = MEAN_IMAGE_BGR
        ### search cube
        ##if datasetName == 'Tsinghua':
            ##self.d = 0.25 # the distance between adjacent points
            ##x_=np.arange(-5,5,self.d)
            ##y_=np.arange(-5,5,self.d)
            ##z_=np.arange(10,30,self.d) #800
        ##elif datasetName == 'MVS':
            ##self.d = 8 #8 # the distance between adjacent points
            ##x_=np.arange(-400,400,self.d) #400 2views_in_group
            ##y_=np.arange(-400,400,self.d)
            ##z_=np.arange(400,1000,self.d) #400,1000

        if xyz_np is None:
            x,y,z = np.meshgrid(x_,y_,z_)
        else:
            x,y,z = xyz_np[:,0], xyz_np[:,1], xyz_np[:,2]
        # xyz1_voxel_candidate/xyz.shape = (N,4) because of homogeneous coordinate
        self.xyz1_voxel_candidate = np.concatenate((x[:,None],y[:,None],z[:,None],np.ones((x.size,1))),axis=1).astype(np.float32)
        self.xyzrgba_selected = np.array([], dtype=np.float32) # used to store the selected points
        self.xyz = None # all the operation is did based on xyz parameter
        self.N_views = 2
        self.patch_size = patch_size
        self.scope_margin = patch_size/2
        self.lightIndx = 0
        # load the camera parameters
        if datasetName == 'Tsinghua':
            self.load_camera_pos_Tsinghua(modelIndx)
            # load the rectified imgs
            # in the list: rectified_img_list[viewIndx-1][light_cond]
            self.load_imgs_Tsinghua(modelIndx)
        elif datasetName == 'MVS':
            self.load_camera_pos_MVS()
            # self.load_imgs_MVS(modelIndx)
            self.load_imgs_MVS_meanIMG(modelIndx)
            self.lightIndx = 0
            self.view_groups_smaller = [range(1,6),range(11,5,-1),range(12,20),range(28,19,-1),\
                range(29,39),range(49,38,-1)]#, [15,16,17,23,24,25,26,32,33,34,35,36]]
            
            
            self.views_4_reconstr = views_4_reconstr
            self.N_views = len(self.views_4_reconstr)
            ##self.view_pairs = [ [self.views_4_reconstr[i], self.views_4_reconstr[i+1]] 
                                            ##for i in range(len(self.views_4_reconstr)-1)]
            ##self.view_pairs.extend([[1,12],[2,11],[3,9],[4,8],[6,19],[7,18],[8,17],[9,16],[10,15],[11,13],[12,29],\
                                    ##[13,28],[14,26],[15,25],[16,24],[17,23],[18,22],[19,21],[20,38],[22,37],[23,35],\
                                    ##[24,34],[26,33],[27,31],[28,30],[30,49],[31,48],[32,46],[33,44],[34,43],[35,42],\
                                    ##[37,41],[38,40]])
            
            ##all_pairIndx = []
            ##for _pairIndx in itertools.combinations(self.views_4_reconstr,2):
                ##all_pairIndx.append(_pairIndx)
            ##random.shuffle(all_pairIndx)  
            self.view_pairs = all_viewPair_viewIndx##all_pairIndx
            
            
            ##self.views_4_reconstr = range(29,39)
            ##self.N_views = len(self.views_4_reconstr)
            ##self.view_pairs = [ [self.views_4_reconstr[i], self.views_4_reconstr[i+1]] 
                                                        ##for i in range(len(self.views_4_reconstr)-1)]            
            
            
            self.viewPairs_viewIndx_array = np.asarray(self.view_pairs)
            self.NOofViewPairs = len(self.view_pairs)
            # viewPairs_viewIndx_array stores the indx of each view in the pairs
            ##self.viewPairs_dimIndx_array = np.asarray([self.views_4_reconstr.index(self.viewPairs_viewIndx_array.reshape(-1,)[i]) 
                                          ##for i in range(self.NOofViewPairs * 2)]).reshape(-1,2)
            self.viewPairs_dimIndx_array = all_viewPair_dimIndx_np                            
        self.NOofScales = 3
        self.size_0Padding = int(self.scope_margin / (0.5 ** self.NOofScales))
        self.pad_zeroes(width = self.size_0Padding)
        self.downSample_imgs(NOofScales = self.NOofScales)
        self.gray_range_thresh=0.90
        self.hue_var_thresh=1/4.
        print(' N_views: {} \n views_4_reconstr: {} \n NOofViewPairs: {} \n color_filter: gray range thresh {:.3f} / hue variance thresh {:.3f}'.\
              format(self.N_views, self.views_4_reconstr, self.NOofViewPairs, \
                     self.gray_range_thresh, self.hue_var_thresh))    

    def reload_xyz_candidate(self):
        self.xyz = np.copy(self.xyz1_voxel_candidate) # reload the voxel candidate

    def load_camera_pos_MVS(self):
        pos_folder = '/home/mengqi/dataset/MVS/pos/'
        self.camera_pos = [[]] # store the camera matrix from the second element, because no 000.txt file
        for i in range(1,65):
            camera_pos_file = pos_folder+'pos_'+str(i).zfill(3)+'.txt'
            with open(camera_pos_file) as f:
                lines = f.readlines()
                self.camera_pos.append(np.asarray([l.split(' ') for l in lines],dtype=np.double))

    def load_camera_pos_Tsinghua(self, modelIndx):
        self.camera_pos = [] # store the camera matrix from the second element, because no 000.txt file
        camera_pos_file = '/home/mengqi/dataset/Tsinghua/download/calibParams'+modelIndx+'.txt'
        with open(camera_pos_file) as f:
            lines = f.readlines()
            for i in range(0, len(lines)/9): # 9 lines for one camera
                Pintra = np.asarray([l.split('\t') for l in lines[i*9+1:i*9+4]],dtype=np.double)
                Pinter = np.asarray([l.split('\t') for l in lines[i*9+5:i*9+8]],dtype=np.double)
                self.camera_pos.append(Pintra.dot(Pinter))

    def load_imgs_MVS(self, modelIndx):
        #preload all the rectified imgs of the model: modelIndx
        rectified_img_folder = '/home/mengqi/dataset/MVS/Rectified/scan'+str(modelIndx)+'/'
        file_list = os.listdir(rectified_img_folder)
        if len(file_list)%8 != 0:
            print('for each view there are EIGHT imgs on different light conditions')
        # if the viewIndx is start from 1, the 0th row of rectified_img_list = []
        self.rectified_img_list = [[None for _ in range(8)] for _ in range(len(file_list)/8+1)]
        for file in file_list:
            viewIndx = int(file.split('_')[1])
            if file.split('_')[2] == 'max.png':
                light_cond = 7
            else:
                light_cond = int(file.split('_')[2])
            self.rectified_img_list[viewIndx][light_cond] = misc.imread(rectified_img_folder+ file)
        img_shape = self.rectified_img_list[1][0].shape
        self.img_scope_wh = [img_shape[1],img_shape[0]]
        print('rectified imgs are loaded')

    def load_imgs_MVS_meanIMG(self, modelIndx):
        #preload all the rectified imgs of the model: modelIndx yield self.xyz1_voxel_candidate[start_indx: start_indx + NO_of_points, :3]

        rectified_img_folder = os.path.join( model_imgs_fld, 'scan'+str(modelIndx))
        file_list = os.listdir(rectified_img_folder)
        # if the viewIndx is start from 1, the 0th row of rectified_img_list = []
        self.rectified_img_list = [[None for _1 in range(1)] for _2 in range(len(file_list)+1)]
        light_cond = 0
        for i, viewIndx in enumerate(view_set):
            img_path = os.path.join(rectified_img_folder, 'rect_{:03}.png'.format(viewIndx))
            self.rectified_img_list[viewIndx][light_cond] = misc.imread(img_path)
            if i == 0:
                img_shape = self.rectified_img_list[viewIndx][0].shape
                self.img_scope_wh = [img_shape[1],img_shape[0]]
        print('rectified imgs are loaded from Voxel_in_Hull.py: ' + rectified_img_folder)

    def pad_zeroes(self, width):
        for each_img in self.rectified_img_list:
            if each_img[0] is None:
                continue         
            each_img[0] = np.lib.pad(each_img[0], ((width,width), (width,width), (0,0)), 'constant', constant_values=0)

    ### after this method, the downsampled imgs will be append to the img-list of each element in the rectified_img_list
    def downSample_imgs(self, NOofScales = 2):
        for each_img in self.rectified_img_list:
            if each_img[0] is None:
                continue
            img_orig = Image.fromarray(each_img[0])
            img_pre_downsample = img_orig
            for i in range(NOofScales):
                img_downsampled = img_pre_downsample.resize(tuple(x/2 for x in img_pre_downsample.size), \
                                                            resample=Image.ANTIALIAS)
                each_img.append(np.array(img_downsampled, dtype=np.uint8))
                img_pre_downsample = img_downsampled
                
        
        
    def load_imgs_Tsinghua(self, modelIndx):
        #preload all the rectified imgs of the model: modelIndx
        rectified_img_folder = '/home/mengqi/dataset/Tsinghua/download/lk'+modelIndx+'/'
        file_list = os.listdir(rectified_img_folder)

        self.rectified_img_list = [[None]]*len(file_list)
        for file in file_list:
            viewIndx = int(file.split('_')[1].split('.')[0])
            self.rectified_img_list[viewIndx][0] = misc.imread(rectified_img_folder+ file)
        img_shape = self.rectified_img_list[0][0].shape
        self.img_scope_wh = [img_shape[1],img_shape[0]]
        self.lightIndx = 0
        print('Tsinghua imgs are loaded')

    # project each line of xyz array to all the views, record the projection position
    # and filter out out-scope ones
    def keep_voxel_in_hull(self):
        self.view_wh = np.zeros((self.xyz.shape[0], self.N_views*2), dtype=np.int) # store the image uv coordinates on each view for each 3d point
        for i, viewIndx in enumerate(self.viewIndxes):
            # project all the 3D point in the xyz array to each view's camera plane, only preserve the ones
            # whose projection is in scope of that view
            perspective = np.dot(self.camera_pos[viewIndx], self.xyz.T)
            perspective = np.divide(perspective, perspective[2,:])
            perspective = perspective[:2,:].T
            # w/h_inscope array of bool
            w_inScope = (perspective[:,0] < self.img_scope_wh[0]-self.scope_margin) & (perspective[:,0] > self.scope_margin)
            h_inScope = (perspective[:,1] < self.img_scope_wh[1]-self.scope_margin) & (perspective[:,1] > self.scope_margin)
            indx_inscope = w_inScope & h_inScope
            #indx_inscope = np.concatenate((w_inScope[:,None],h_inScope[:,None], \
                                   #(perspective > 1+self.scope_margin)),axis=1).all(axis=1)

            self.view_wh[:,i*2:(i+1)*2] = perspective.astype(np.int)
            # make sure the view_wh have the same number of rows with xyz array
            self.view_wh = self.view_wh[indx_inscope]
            self.xyz = self.xyz[indx_inscope]
            
    # input: xyz1, array of n 4D points
    # store: view_wh: an n*v*3 array, 1st channel indicating whether nth point is in the scope of vth view
    #                         2nd/3rd channel is the view_wh information
    def proj_pt_2_views(self, xyz1):
        view_wh = np.zeros((xyz1.shape[0], self.N_views, 3), dtype=np.int) 
        for _dimIndx, viewIndx in enumerate(self.views_4_reconstr):
            # project all the 3D point in the xyz array to each view's camera plane, only preserve the ones
            # whose projection is in scope of that view
            perspective = np.dot(self.camera_pos[viewIndx], xyz1.T)
            perspective = np.divide(perspective, perspective[2,:])
            perspective = perspective[:2,:].T
            # w/h_inscope array of bool
            w_inScope = (perspective[:,0] < self.img_scope_wh[0]) & (perspective[:,0] >= 0)
            h_inScope = (perspective[:,1] < self.img_scope_wh[1]) & (perspective[:,1] >= 0)
            indx_inscope = w_inScope & h_inScope
            #indx_inscope = np.concatenate((w_inScope[:,None],h_inScope[:,None], \
                                   #(perspective > 1+self.scope_margin)),axis=1).all(axis=1)
    
            view_wh[:,_dimIndx,1:3] = perspective.astype(np.int)
            view_wh[:,_dimIndx,0] =indx_inscope 
        return view_wh
            
        
    # proj_pt_2_views will store the in-scope information in the 1st channel of the param: view_wh
    # if a point is outscope of the ith view, the view-pairs, like [i, j]/[j, i], including the ith view will be ignored
    # after which, the view_wh's 1st channel will only store the views which are in the corresponding valid view pairs
    # also save the valid view pairs information in the self.valid_view_pairs
    def keep_valid_view_pairs(self, view_wh):
        pt_inScope_in_viewIndxes = view_wh[:,:,0]
        return pt_inScope_in_viewIndxes[:,self.viewPairs_dimIndx_array[:,0]] & pt_inScope_in_viewIndxes[:,self.viewPairs_dimIndx_array[:,1]]
            
    # based on the self.valid_view_pairs and the self.view_pairs
    # mark the views which are need to be used to generate patches passing through the network
    def get_valid_views(self, valid_view_pairs):
        # valid_views will indicate whether the patch in the ith view of the nth point is useful for further calculation
        valid_views = np.zeros((valid_view_pairs.shape[0], self.N_views)).astype(np.bool)
        for i in range(self.NOofViewPairs):
            view_pair_i = np.zeros(valid_views.shape).astype(np.bool)
            view1_indx = self.viewPairs_dimIndx_array[i,0]
            view2_indx = self.viewPairs_dimIndx_array[i,1]
            view_pair_i[:,view1_indx] = valid_view_pairs[:,i]
            view_pair_i[:,view2_indx] = valid_view_pairs[:,i]
            valid_views |= view_pair_i
        return valid_views
    
    def get_proper_patch_from_img(self, img, patch_r, w, h, rgb_var_threshold, patch_r_enlarged=0):
        if patch_r_enlarged == 0: # the first view in the current view group
            patch_tmp = img[h-patch_r: h+patch_r, w-patch_r: w+patch_r]
            rgb_var = np.var(patch_tmp.reshape(-1,3), axis=0).sum()/3
            
            if rgb_var > rgb_var_threshold: # for most of the cases
                return [patch_r, patch_tmp]
            else:
                img_PIL = Image.fromarray(img)
                patch_r_Large = patch_r * 2
                for i in range(3):
                    # PIL.crop: 4-tuple defining the left, upper, right, and lower pixel coordinate, can access out of img range
                    patch_Large = img_PIL.crop((w-patch_r_Large, h-patch_r_Large, w+patch_r_Large+1, h+patch_r_Large+1))            
                    patch_tmp = np.array(patch_Large.resize((patch_r*2, patch_r*2), Image.ANTIALIAS),dtype=np.uint8)
                    rgb_var = np.var(patch_tmp.reshape(-1,3), axis=0).sum()/3
                    if rgb_var > rgb_var_threshold:
                        break
                    else:
                        patch_r_Large *= 2
                return [patch_r_Large, patch_tmp]
        elif patch_r_enlarged == patch_r:
            patch_tmp = img[h-patch_r: h+patch_r, w-patch_r: w+patch_r]
            return [patch_r, patch_tmp]
        else:
            img_PIL = Image.fromarray(img)
            patch_Large = img_PIL.crop((w-patch_r_enlarged, h-patch_r_enlarged, w+patch_r_enlarged+1, h+patch_r_enlarged+1))   
            patch_tmp = np.array(patch_Large.resize((patch_r*2, patch_r*2), Image.ANTIALIAS),dtype=np.uint8)
            return [patch_r_enlarged, patch_tmp]
    
    # this will replace the method 'load_dataset' in the 'data_reader.py'
    def generate_voxel_patches(self):
        patch_r = self.patch_size/2
        # inputs.size = (N * Noofviews, 3, 32, 32)
        self.inputs = np.zeros((self.xyz.shape[0] * self.N_views,3,self.patch_size,self.patch_size),\
                          dtype=np.float32) # inputs[i][j]: the ith set jth view
        for i in range(0,self.xyz.shape[0]):
            patch_r_enlarged = 0
            for j, viewIndx in enumerate(self.viewIndxes):
                w,h = self.view_wh[i,j*2:(j+1)*2].round()
                # rectified_img_list[viewIndx][light_cond]
                img=self.rectified_img_list[viewIndx][self.lightIndx]
                # path is not centered around (w,h), not good,
                # but the Image.resize takes long time
                # and CNN needs even size because of pooling layer
                patch = img[h-patch_r: h+patch_r, w-patch_r: w+patch_r]
                ##[patch_r_enlarged, patch] = self.get_proper_patch_from_img(img, patch_r, w, h, rgb_var_threshold=200, patch_r_enlarged = patch_r_enlarged)
                # preprocess the img for VGG
                # Shuffle axes to c01: channel,height,weigth
                patch = np.swapaxes(np.swapaxes(patch, 1, 2), 0, 1)
                # Convert to BGR
                patch = patch[::-1, :, :]
                #patch = patch - self.MEAN_IMAGE_BGR[:,None,None]
                self.inputs[i*self.N_views + j] = patch

    ## TODO: latter on, we can firstly iterate w.r.t. the view img, and generate all the patches for all the feasible points, 
    ## Then change to another view image. This may speed up the patch prepare stage.
    def generate_patches_rgb_from_valid_views(self, iteration):
        patch_r = self.patch_size/2
        # inputs.size = (N * Noofviews, 3, 32, 32)
        (indx_n,indx_v) = np.where(self.valid_views)
        NOofValidPatches = self.valid_views.sum()
        self.inputs = np.zeros((NOofValidPatches,3,self.patch_size,self.patch_size),\
                          dtype=np.float32) # inputs[i][j]: the ith set jth view
        inputs_selection = np.ones((NOofValidPatches),dtype=np.bool)
        self.pt_rgb = np.zeros((self.NOofPts, self.N_views, 3), dtype=np.uint8)
        if iteration < self.NOofScales:
            img_indx = self.NOofScales - iteration
        else:
            img_indx = 0
        for i in range(NOofValidPatches):
            w,h = ((self.view_wh[indx_n[i], indx_v[i], 1:3].round() + self.size_0Padding)* self.scale_wh).astype(int)
            img=self.rectified_img_list[self.views_4_reconstr[indx_v[i]]][img_indx]
            # path is not centered around (w,h), not good,
            # but the Image.resize takes long time
            # and CNN needs even size because of pooling layer
            patch = img[h-patch_r: h+patch_r, w-patch_r: w+patch_r]
            self.pt_rgb[indx_n[i],indx_v[i],:] = patch[patch_r, patch_r]
            ##[patch_r_enlarged, patch] = self.get_proper_patch_from_img(img, patch_r, w, h, rgb_var_threshold=200, patch_r_enlarged = patch_r_enlarged)
            # preprocess the img for VGG
            # Shuffle axes to c01: channel,height,weigth
            patch = np.swapaxes(np.swapaxes(patch, 1, 2), 0, 1)
            # Convert to BGR
            patch = patch[::-1, :, :]
            #patch = patch - self.MEAN_IMAGE_BGR[:,None,None]
            self.inputs[i] = patch
            # if this patch is near pure color patch, ignore this point, because it may be a background point
            rgb_max = patch.max(axis=-1).max(axis=-1)
            rgb_min = patch.min(axis=-1).min(axis=-1)
            rgb_max_channel_range = (rgb_max - rgb_min).max()
            if rgb_max_channel_range < 18.: ## a gray_range_threshold
                self.valid_views[indx_n[i]] = 0 ## delete this 3D point
                inputs_selection[np.where(indx_n==indx_n[i])] = 0 ##remember to delete the corresponding 3D pts in the inputs
            
        self.inputs = self.inputs[inputs_selection]
                
                
    def iterate_minibatches(self, batchsize):
        end_idx = 0
        for start_indx in range(0, self.xyz.shape[0] - batchsize + 1, batchsize):
            end_idx = start_indx
            yield self.inputs[self.N_views * start_indx: self.N_views * (start_indx + batchsize)], \
                  self.xyz[start_indx: start_indx + batchsize, :3]
        end_idx += batchsize
        yield self.inputs[self.N_views * end_idx: ], self.xyz[end_idx: , :3]

    def keep_highProb_voxel(self, test_labels_stack, threshold):
        # todo the size of the probability_stack is size of multiple of the batch_size (not good)
        # filter = np.concatenate(((test_labels_stack[:,0] > threshold)[:,None],(test_labels_stack[:,1:] > 0.4)), axis=1).any(axis=1)
        filter = (test_labels_stack[:,0] > threshold)
        self.xyz = self.xyz[filter]



    def stack_highProb_xyzrgba(self, xyzrgba, test_labels_stack, thresholdmin, thresholdmax):
        filter = (test_labels_stack[:,0] < thresholdmax) & (test_labels_stack[:,0] >= thresholdmin)  # > 0.1 may filter out the pure color patches
        print('stack_highProb_xyzrgba: {}/{}'.format(sum(filter),filter.size))
        xyzrgba = xyzrgba[filter]
        self.xyzrgba_selected = np.vstack([self.xyzrgba_selected, xyzrgba]) if self.xyzrgba_selected.size else xyzrgba

    def filter_selectedxyz_xyzcandidate(self, xyzrgba, test_labels_stack, thresholdmin):
        filter = test_labels_stack[:,0] > thresholdmin  
        xyzrgba_delet = xyzrgba[filter]
        print('before filter_selectedxyz_xyzcandidate, size = {} , {}'.format(self.xyz1_voxel_candidate.shape[0], 
                                                                              self.xyzrgba_selected.shape[0]))
        keepindx_xyzrgba_selected = np.ones((self.xyzrgba_selected.shape[0],), dtype=np.bool)
        keepindx_xyz1_voxel_candidate = np.ones((self.xyz1_voxel_candidate.shape[0],), dtype=np.bool)
        for r in range(0, xyzrgba_delet.shape[0]):
            keepindx_xyzrgba_selected &= ~((self.xyzrgba_selected[:,:3] == xyzrgba_delet[r,:3]).all(axis=1))
            keepindx_xyz1_voxel_candidate &= ~((self.xyz1_voxel_candidate[:,:3] == xyzrgba_delet[r,:3]).all(axis=1))
            
        self.xyzrgba_selected = self.xyzrgba_selected[keepindx_xyzrgba_selected]
        self.xyz1_voxel_candidate = self.xyz1_voxel_candidate[keepindx_xyz1_voxel_candidate]
        print('after filter_selectedxyz_xyzcandidate, size = {} , {}'.format(self.xyz1_voxel_candidate.shape[0], 
                                                                              self.xyzrgba_selected.shape[0]))


    # gray_range_thresh, hue_var_thresh in the range [0,1]
    def color_filter(self, rgb_array, mask, gray_range_thresh, hue_var_thresh):
    
        gray_array = np.ma.array(rgb_array.mean(axis=-1), mask=mask)
        gray_array_range = gray_array.max(axis=1) - gray_array.min(axis=1)
        
        # ignore the black & white color when dealing with Hue, because
        mask_gray = (rgb_array[:,:,0] == rgb_array[:,:,1]) & (rgb_array[:,:,2] == rgb_array[:,:,1])
        
        hsv_array = matplotlib.colors.rgb_to_hsv(rgb_array/255.)
        # ATTENTION: mask in np.ma means: ignored(TRUE), reserved(FALSE)
        hsv_hue_1 = np.ma.array(hsv_array[:,:,0], mask=mask&mask_gray)
        hsv_hue_std_1 = np.ma.std(hsv_hue_1, axis=1)
        hsv_array_tmp = np.copy(hsv_array[:,:,0])
        # let the hue value turn 180d on the hue circle, calculate the var again, take the minimum one
        hsv_array_tmp[hsv_array[:,:,0]>0.5] -= 0.5
        hsv_array_tmp[hsv_array[:,:,0]<0.5] += 0.5
        hsv_hue_2 = np.ma.array(hsv_array_tmp, mask=mask)
        hsv_hue_std_2 = np.ma.std(hsv_hue_2, axis=1)
        hsv_hue_std = np.minimum(hsv_hue_std_1, hsv_hue_std_2)              
        
        return ((gray_array_range < (gray_range_thresh*255.)) & (hsv_hue_std < hue_var_thresh)).astype(np.bool)
        
            
    def select_xyzrgba(self, diff_X_Euclid, threshold):
        diff_X_Euclid[diff_X_Euclid < 0.1] = threshold[1]+100 # in order to filter out the point out of the visual hull
        on_surf_decision = diff_X_Euclid < threshold[1]
        diff_X_Euclid[diff_X_Euclid > threshold[1]] = 0
        # if there are more than NOofViewPairs/8 on-surface decisions  
        # self.NOofViewPairs / 3
        # self.valid_view_pairs.sum(axis=1) / 3
        # np.maximum( self.valid_view_pairs.sum(axis=1) / 3, 1 )
        filter = (on_surf_decision.sum(axis=1) > (self.valid_view_pairs.sum(axis=1) / 7.)).astype(np.bool) 
        
        filter_color = self.color_filter(self.pt_rgb[:,self.viewPairs_dimIndx_array[:,0],:], ~on_surf_decision, \
                     gray_range_thresh=self.gray_range_thresh, hue_var_thresh=self.hue_var_thresh)
        
        #filter &= filter_color
        
        #on_surf_decision[~filter] = 0
        ###diff_X_Euclid_argmin = np.zeros(on_surf_decision.shape, dtype=np.bool)
        # only select the argmin in each row
        ###diff_X_Euclid_argmin[np.c_[np.arange(diff_X_Euclid.shape[0]),\
        ###                           diff_X_Euclid.argmin(axis=1)].T.tolist()] = 1
        # calculate the pt's rgb, use masked mean method in numpy
        pt_view_pair_rgb = self.pt_rgb[:,self.viewPairs_dimIndx_array[:,0],:] 
        #pt_view_pair_rgb = (self.pt_rgb[:,self.viewPairs_dimIndx_array[:,0],:] + self.pt_rgb[:,self.viewPairs_dimIndx_array[:,1],:])/2
        #pt_view_pair_mask = on_surf_decision.astype(np.bool)
        ###pt_view_pair_mask = diff_X_Euclid_argmin.astype(np.bool)
        # ATTENTION: mask in np.ma means: ignored(TRUE), reserved(FALSE)
        pt_rgb_masked = np.ma.array(pt_view_pair_rgb, dtype=np.uint8, copy=False, order=False, \
                                    mask= ~on_surf_decision[:,:,None].repeat(3,axis=2)) # on_surf_decision / pt_view_pair_mask
        RGB = pt_rgb_masked.mean(axis=1).data.astype(np.uint8) # along the view axis
        # embed the number of on_surf_decision into the alpha channel
        A = 254##-self.NOofViewPairs+on_surf_decision.sum(axis=1)
        rgba = np.zeros((self.NOofPts,),dtype=np.float32)
        for i in range(self.NOofPts):
            # thanks to: 'http://www.pcl-users.org/How-to-convert-from-RGB-to-float-td4022278.html'
            rgba[i] = struct.unpack('f', chr(RGB[i,2]) + chr(RGB[i,1]) + chr(RGB[i,0]) + chr(A))[0]
        self.rgba = rgba
        self.xyzrgba = np.c_[self.xyz,rgba]
        
        
        self.xyzrgba_selected = np.vstack([self.xyzrgba_selected, self.xyzrgba[filter]]) if self.xyzrgba_selected.size else self.xyzrgba[filter]
        return filter
        

    def stack_highProb_pixel2voxel(self, xyzrgba, test_labels_stack, threshold, topMax = 1):

        filter_pixel2voxel = np.zeros((test_labels_stack.shape[0], 2),dtype=np.int8) # 2 colomns for L/R views
        for j, viewIndx in enumerate(self.viewIndxes):
            #if j != 1:
                #continue
            pt_w = self.view_wh[:,j*2]
            pt_h = self.view_wh[:,j*2+1]
            # test_labels_stack maller means better~
            pt_Probability = test_labels_stack[:,0] * -1
            sortedIndx_whP = np.lexsort((pt_Probability,pt_h,pt_w))

            for n, indx in enumerate(sortedIndx_whP[::-1]): # because the last one is the maximum
                if n == 0:
                    pt_counter4_WiHi = 0
                    pre_w = pt_w[indx]
                    pre_h = pt_h[indx]
                if pt_Probability[indx] < threshold * -1:
                    continue
                if pt_w[indx]==pre_w and pt_h[indx]==pre_h:
                    if pt_counter4_WiHi < topMax:
                        pt_counter4_WiHi += 1
                        filter_pixel2voxel[indx, j] = 1
                    else:
                        continue
                else:
                    pre_w = pt_w[indx]
                    pre_h = pt_h[indx]
                    filter_pixel2voxel[indx, j] = 1
                    pt_counter4_WiHi = 1

        filter_pixel2voxel_LR = filter_pixel2voxel[:,0] & filter_pixel2voxel[:,1] # ONLY the point which is chosen by the 2 views at the same time will be left
        print('stack_highProb_pixel2voxel: {}/{}'.format(sum(filter_pixel2voxel_LR),(test_labels_stack[:,0]<threshold).sum()))
        #self.xyz = self.xyz[filter_pixel2voxel.astype(np.bool)]
        xyzrgba = xyzrgba[filter_pixel2voxel_LR.astype(np.bool)]
        self.xyzrgba_selected = np.vstack([self.xyzrgba_selected, xyzrgba]) if self.xyzrgba_selected.size else xyzrgba

    def remove_duplicate_of_xyzrgba_selected(self):
        xyzrgba_selected_ = self.xyzrgba_selected
        filter = np.zeros((xyzrgba_selected_.shape[0],), dtype=np.bool)
        sorted_indx = np.lexsort([xyzrgba_selected_[:,0], xyzrgba_selected_[:,1], xyzrgba_selected_[:,2]])
        pre_true_index = sorted_indx[0]
        filter[pre_true_index] = 1
        for i in range(1,sorted_indx.shape[0]):
            if (xyzrgba_selected_[sorted_indx[i], :3] == xyzrgba_selected_[pre_true_index, :3]).all():
                continue
            else:
                pre_true_index=sorted_indx[i]
                filter[sorted_indx[i]] = 1
        self.xyzrgba_selected = self.xyzrgba_selected[filter]
        print('remove_duplicate_of_xyzrgba_selected, before/after pts: {} / {}'.format(filter.sum(), filter.shape[0]))

    def OctExpand_voxels(self):
        r = self.d / 4
        # neighbor.shape = 8*4
        neighbor = np.asarray(np.meshgrid([r,-1*r],[r,-1*r],[r,-1*r],[0])).reshape((4,-1)).T
        xyz1_tmp = np.copy(self.xyzrgba_selected)
        xyz1_tmp[:,-1] = 1 ## NOTICE that xyzrgba_selected only has 4 elements in each row, (rgba is converted into one float)
        self.xyz1_voxel_candidate = (xyz1_tmp[:,None,:]-neighbor[None,:,:]).reshape((-1,4))
        #self.xyz1_voxel_candidate[:,:3] += (self.d / 10) * (np.random.random((self.xyz1_voxel_candidate.shape[0],3))-0.5)
        self.d /= 2

    def iterate_viewGroups(self):
        if self.datasetName == 'MVS':
            # make sure the list element of the view_groups_valid has enough members.
            view_groups_valid = [v_i for v_i in self.view_groups_smaller if len(v_i)>=self.N_views]
            print ('now, only choose 2 views from each view_group.')
            for view_group_selected in view_groups_valid:
                indx_L_view = random.choice(range(0,len(view_group_selected)-1))
                #indx_NOofviews = random.sample(range(0,len(view_group_selected)),self.N_views)
                indx_NOofviews = [indx_L_view, indx_L_view+1] # ordered element
                # use the indx_NOofviews as the indices of the list
                # view only stores the views which are really used for each sample (say 4-views)
                yield [view_group_selected[indx_i] for indx_i in indx_NOofviews]

    def iterate_2views_in_group(self):
        # the views we used to reconstruct the surface 
        if self.datasetName == 'MVS':
            view_group = range(11,5,-1)[::-1] #(11,5,-1)
        elif self.datasetName == 'Tsinghua':
            view_group = range(0,7) #range(0,12)
        print ('the views are {}'.format(view_group))
        for vi in range(0, len(view_group)-1):
            yield [view_group[vi], view_group[vi+1]]

    def iterate_candidates(self, candidates_valid_views, Max_img_patches):
        start_indx = 0
        candidates_cumsum_valid_views = np.cumsum(candidates_valid_views.sum(axis=1))
        end_idx = 0
        _N = candidates_valid_views.shape[0]
        print('processing {} points: '.format(_N))
        while start_indx < _N:
            if start_indx == 0:
                NO_of_points = (candidates_cumsum_valid_views[start_indx:] < \
                                            Max_img_patches).sum()       
            else:
                NO_of_points = (candidates_cumsum_valid_views[start_indx:] - candidates_cumsum_valid_views[start_indx-1] < \
                                Max_img_patches).sum()
            print '{}'.format(start_indx), 
            yield start_indx, start_indx+NO_of_points # return the pt_indx_range
            ##if (start_indx+NO_of_points) > _N:
                ##yield 
            start_indx += NO_of_points
            
        
        #for start_indx in range(0, self.xyz1_voxel_candidate.shape[0] - NO_of_points + 1, NO_of_points):
            #end_idx = start_indx
            #yield self.xyz1_voxel_candidate[start_indx: start_indx + NO_of_points, :3]
        #end_idx += NO_of_points
        #yield self.xyz1_voxel_candidate[end_idx: , :3]        

# v = voxel_in_hull('G', 0,range(0,12),[1,1,1],32)
# v = voxel_in_hull('MVS',3, 0,[10,11,12,13],np.array([1,1,1]),32)
# v.keep_voxel_in_hull()
# v.generate_voxel_patches()
# for view_indexes in v.iterate_viewGroups():
#     print(view_indexes)
# print('ll ')

def save_pcd(xyzrgba_stack, model_folder, filename):
    f = open(model_folder+filename,'w')
    header = """# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {0}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {0}
DATA ascii\n""".format(xyzrgba_stack.shape[0])
    #np.save(open(save_dataset_folder+str(modelIndx).zfill(3)+'_'+mode+'_xyz.data', "w" ), xyz)
    f.write(header)
    for l in range(0,xyzrgba_stack.shape[0]):
        f.write('{} {} {} {}\n'.format(xyzrgba_stack[l,0],xyzrgba_stack[l,1],xyzrgba_stack[l,2], xyzrgba_stack[l,3]))
    f.close()
    print("save {} points to: {}{}".format(xyzrgba_stack.shape[0], model_folder,filename))
