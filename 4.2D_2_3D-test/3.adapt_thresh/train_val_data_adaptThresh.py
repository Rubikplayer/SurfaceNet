import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as fi
import os
import time
import scipy
import random
import struct
import itertools

##from VGG_Triplet import *
##from Voxel_in_Hull import voxel_in_hull
##from Voxel_in_Hull import save_pcd

random.seed(201605)

pts_in_voxel_MAX = 10. # so use np.uint8 to save the train/val data file
pts_in_voxel_MIN = 0.
downsample_scale = 5
grids_d = 50 # NO of voxels resolution in each axis
D = grids_d
D_randcrop = 32##5*D/6
D_center = 26 ##20, 26 may speed up the process
N = grids_d ** 3
N_cubeParams = 5 # x,y,z,resol,modelIndx


VGG_triplet_thresh = [None,6.5] ## if the two patches are too similiar, they are still be ignored. Because the baseline may be not large enough.

train_ON = False
val_ON = False
reconstr_ON = True

train_with_normal = False
visualize_when_generateData_ON = False

# if want to change the train/val_set, remember to delete the data files
train_set = range(1,17)
val_set = [17]+range(109,111)
test_set = [17]

if reconstr_ON:
    train_ON, val_ON = False, False
    D = D_randcrop ## because no random_crop, the initial size of the cube should be DXDXD
    N_cubeParams = 8 # x,y,z,resol,modelIndx,indx_d0,indx_d1,indx_d2
    N_views = 49
    view_set = random.sample(range(1,50),N_views) ## 1-49
    

temp_folder = '/hdd1t/dataset/MVS/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D_normal/temp_visualization/'
#visualizer = '/home/mji/pcl/pcl_program/tutorial/20160522_01_pcl_multi_visualizers_XYZRGBT_backup/build_qt/pcl_visualizer_demo'
visualizer = '/home/mji/pcl/pcl_program/tutorial/20160725_pcl_multi_visualizers_XYZRGBT_normal-backup/build-qt/pcl_visualizer_demo'
data_fld = '/hdd1t/dataset/MVS/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/'
voxelVolume_txt_fld = '/hdd1t/dataset/MVS/samplesVoxelVolume/pcl_txt_50x50x50_2D_2_3D_normal/'
camera_po_txt_fld = '/home/mengqi/dataset/MVS/pos/'
model_imgs_fld = '/hdd1t/dataset/MVS/Rectified_mean/'

pcd_folder = '/hdd1t2/dataset/MVS/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/saved_pcd/'
pcl_pcd2ply_exe = '~/Downloads/pcl-trunk/build/bin/pcl_pcd2ply'
       
                
all_viewPair_viewIndx = []
for _pairIndx in itertools.combinations(view_set,2):
    all_viewPair_viewIndx.append(_pairIndx)
random.shuffle(all_viewPair_viewIndx) ## So that, latter on, select the first N == randomly select N
all_viewPair_viewIndx_np = np.asarray(all_viewPair_viewIndx, dtype=np.uint8) 

dict_viewIndx_2_dimIndx = {_viewIndx:_dimIndx for _dimIndx, _viewIndx in enumerate(view_set)}
dict_dimIndx_2_viewIndx = {_dimIndx:_viewIndx for _dimIndx, _viewIndx in enumerate(view_set)}
map_viewIndx_2_dimIndx_np = lambda x: np.asarray(map(dict_viewIndx_2_dimIndx.get, x.flatten())).reshape(x.shape)
map_dimIndx_2_viewIndx_np = lambda x: np.asarray(map(dict_dimIndx_2_viewIndx.get, x.flatten())).reshape(x.shape)

all_viewPair_dimIndx_np = map_viewIndx_2_dimIndx_np(all_viewPair_viewIndx_np)
   
#----------------------------------------------------------------------
def get_VGG_triplet_Net_featureVec(modelIndx,xyz_np,view_set):
    """
    This func is from the reconstruction part of the VGG_triplet_Net{20160313-01-d.2-(0.90,1_4)-Demo}
    return: 
    feature_euclidDiff_embed: np{NOofPts, N_viewPairs}
    ##feature_embed: np{NOofPts, N_views, feature_embed_dim}
    desicion_onSurf: np{NOofPts,} with 0/1 to 
    """

    # voxels = voxel_in_hull(modelIndx = sampleIndx_test, light_condition = 2, \
    #                        viewIndxes=viewIndxes, MEAN_IMAGE_BGR = MEAN_IMAGE_BGR)
    voxels = voxel_in_hull(modelIndx = modelIndx, datasetName = 'MVS', views_4_reconstr = view_set,\
                           MEAN_IMAGE_BGR = MEAN_IMAGE_BGR, patch_size=hw_size, xyz_np = xyz_np)
    
    threshold = VGG_triplet_thresh
    _ = 2 ## previously, there is an iterative process.
    voxels.scale_wh = 0.5 ** max(voxels.NOofScales - _, 0) # 1
    print('\n*****vgg triplet threshold = {}'.format(threshold))
    #np.random.shuffle(voxels.xyz1_voxel_candidate)            
    voxels.candidates_view_wh = voxels.proj_pt_2_views(voxels.xyz1_voxel_candidate)
    #print("2it takes {:.3f}s".format(time.time() - start_time))
    voxels.candidates_valid_view_pairs = voxels.keep_valid_view_pairs(voxels.candidates_view_wh)
    #print("3it takes {:.3f}s".format(time.time() - start_time))
    tmp = np.copy(voxels.candidates_view_wh[:,:,0])
    ## filter out the pts which only have 1 valid view. 
    ## If don't do it, the pts with only 1 valid view will have some bug when diff_X_Euclid: [non_zero_feature - zero_feature]**2 
    tmp[tmp.sum(axis=-1) ==1] = 0 
    voxels.candidates_valid_views = tmp #voxels.get_valid_views(voxels.candidates_valid_view_pairs)  
    feature_embed_list = []
    diff_X_Euclid_list = []
    desicion_onSurf_list = []
    for pt_batch_start_end_indx in voxels.iterate_candidates(voxels.candidates_valid_views, Max_img_patches = 3900):  
        #print("1it takes {:.3f}s".format(time.time() - start_time))
        start_indx, end_indx = pt_batch_start_end_indx
        pt_batch_indx_range = range(start_indx, end_indx)
        voxels.xyz = voxels.xyz1_voxel_candidate[pt_batch_indx_range, :-1]
        voxels.NOofPts = len(pt_batch_indx_range)
        voxels.view_wh = voxels.candidates_view_wh[pt_batch_indx_range]
        voxels.valid_views = voxels.candidates_valid_views[pt_batch_indx_range]
        voxels.valid_view_pairs = voxels.candidates_valid_view_pairs[pt_batch_indx_range]
        #print("4it takes {:.3f}s".format(time.time() - start_time))
        voxels.generate_patches_rgb_from_valid_views(iteration = _)
        #print("5it takes {:.3f}s".format(time.time() - start_time))
        if voxels.inputs.shape[0] == 0:
            continue
        output_feature_embed = test_fn(voxels.inputs)
        #print("6it takes {:.3f}s".format(time.time() - start_time))
        (indx_n,indx_v) = np.where(voxels.valid_views)
        feature_embed = np.zeros((voxels.NOofPts, voxels.N_views, output_feature_embed.shape[1]),dtype=np.float32) ## because the view indx starts from 1.
        ##for i in range(voxels.valid_views.sum()):
            ##feature_embed[indx_n[i], indx_v[i]]=output_feature_embed[i]
        feature_embed[indx_n, indx_v]=output_feature_embed
        #print("7it takes {:.3f}s".format(time.time() - start_time))
        ##feature_embed_list.append(feature_embed)
        diff_X = feature_embed[:,voxels.viewPairs_dimIndx_array[:,0],:] - feature_embed[:,voxels.viewPairs_dimIndx_array[:,1],:] 
        diff_X_sq_sum = (diff_X**2).sum(axis=-1)
        diff_X_Euclid = diff_X_sq_sum ** .5            
        #print("8it takes {:.3f}s".format(time.time() - start_time))          
        filter_sub = voxels.select_xyzrgba(diff_X_Euclid, threshold=threshold) ## diff_X_Euclid will be updated
        diff_X_Euclid_list.append(diff_X_Euclid)
        #print("9it takes {:.3f}s".format(time.time() - start_time))
        desicion_onSurf_list.append(filter_sub)
    return np.vstack(diff_X_Euclid_list), np.hstack(desicion_onSurf_list)



#----------------------------------------------------------------------
def load_all_cameraPO_files_f64(view_list):
    """ 
    load the camera POs into a np.float64
    input: 
    view_list indicate which viewIndx([3,21,53...]) are used.
    output:
    cameraPOs_npf64 {len(view_list),3,4} use the dimIndx(0,1,2...) to access different views
    """
    cameraPOs_npf64 = np.zeros((len(view_list),3,4),dtype=np.float64)
    for _dimIndx, _viewIndx in enumerate(view_list):
        file_name = camera_po_txt_fld+'pos_{:03}.txt'.format(_viewIndx)
        if not os.path.exists(file_name):
            print 'no cameraPO file '+file_name
            continue
        cameraPOs_npf64[_dimIndx] = np.loadtxt(file_name,dtype=np.float64,delimiter = ' ')
    return cameraPOs_npf64

def load_model_meanIMG(modelIndx):
    ##preload all the rectified imgs of the model: modelIndx
    rectified_img_folder = model_imgs_fld+'scan'+str(modelIndx)+'/'
    file_list = os.listdir(rectified_img_folder)
    # if the viewIndx is start from 1, the 0th row of model_imgs = []
    model_imgs = [[None for _ in range(1)] for _ in range(len(file_list)+1)]
    for file in file_list:
        viewIndx = int(file.split('_')[1])
        if file.split('_')[2] == 'mean.jpg':
            light_cond = 0
        model_imgs[viewIndx][light_cond] = scipy.misc.imread(rectified_img_folder+ file)
    ##img_shape = model_imgs[1][0].shape
    ##img_scope_wh = [img_shape[1],img_shape[0]]
    print('rectified imgs of indx: {} are loaded'.format(modelIndx))
    return model_imgs


def load_model_meanIMG_aslist(modelIndx):
    ##preload all the rectified imgs of the model: modelIndx
    rectified_img_folder = model_imgs_fld+'scan'+str(modelIndx)+'/'
    file_list = os.listdir(rectified_img_folder)
    # if the viewIndx is start from 1, the 0th row of model_imgs = []
    model_imgs = [[None for _ in range(1)] for _ in range(len(file_list)+1)]
    for file in file_list:
        viewIndx = int(file.split('_')[1])
        if file.split('_')[2] == 'mean.jpg':
            light_cond = 0
        model_imgs[viewIndx][light_cond] = scipy.misc.imread(rectified_img_folder+ file)
    print('rectified imgs of indx: {} are loaded'.format(modelIndx)) 
    return model_imgs


#----------------------------------------------------------------------
def load_model_meanIMG_asnp(modelIndx):
    ##preload all the rectified imgs of the model: modelIndx
    rectified_img_folder = model_imgs_fld+'scan'+str(modelIndx)+'/'
    model_imgs_np = None
    
    file_list = os.listdir(rectified_img_folder)
    ## if the viewIndx is start from 1, the 0th row of model_imgs = []
    for i,file in enumerate(file_list):
        viewIndx = int(file.split('_')[1])
        img = scipy.misc.imread(rectified_img_folder+ file)
        if i==0:
            model_imgs_np = np.zeros((len(file_list)+1,)+img.shape, dtype=np.uint8)
        model_imgs_np[viewIndx] = img

    model_imgs_npy_file = model_imgs_fld+'npy/{}.npz'.format(modelIndx)
    with open(model_imgs_npy_file,'w') as f:
        ##np.save(f,model_imgs_np.astype(np.uint8) )  ##very fast when save, same speed for loading
        np.savez_compressed(f,model_imgs_np.astype(np.uint8)) ## 2/3 compressed, very slow when save
        print('compressed {}.npz is saved'.format(modelIndx))
    return model_imgs_np


#----------------------------------------------------------------------
def load_model_VGGOccupancy_VGGFeature_asnp(modelIndx, model_densityCube_param):
    """
    similiar to func 'load_modellist_meanIMG' & 'load_train_val_gt_asnp'
    preload all the VGGNet data of the model: modelIndx
    model_densityCube_param contains the all the cubes parameter for this particular modelIndx
    
    return: similiar with the output of the func 'get_VGG_triplet_Net_featureVec'
    desicion_onSurf: np{N_pts,} with 0/1 to 
    feature_embed: np{N_pts, N_views, feature_embed_dim}
    similarity_thresh = threshold
    """
    modelParam = model_densityCube_param
    model_feature_embed, model_desicion_onSurf = get_VGG_triplet_Net_featureVec(modelIndx=modelIndx,\
                xyz_nplist=[i + D*modelParam[:,3]/2 for i in [modelParam[:,0],modelParam[:,1],modelParam[:,2]]])
    model_VGGNet_npy_file = voxelVolume_txt_fld+'VGG_triplet_Net_outputs/{}.npz'.format(modelIndx)
    
    model_feature_embed = model_feature_embed.astype(np.float32)
    model_desicion_onSurf = model_desicion_onSurf.astype(np.uint8)
    
    with open(model_VGGNet_npy_file,'w') as f:
        np.savez(f, model_desicion_onSurf, model_feature_embed)
        print('{} is saved'.format(model_VGGNet_npy_file))
        
    return model_desicion_onSurf, model_feature_embed 
        

#----------------------------------------------------------------------
def load_modellist_meanIMG(modelIndx_list):
    """to index the imgs of perticular nth model, just use modellist_imgs[n]"""
    modellist_imgs = [None for _ in range(max(modelIndx_list)+1)] ## only the position of indexes in the modelIndx_list are non-None
    for n in modelIndx_list:
        model_imgs_npy_file = model_imgs_fld+'npy/{}.npz'.format(n)
        if os.path.exists(model_imgs_npy_file): 
            modellist_imgs[n] = np.load(model_imgs_npy_file)['arr_0']
        else:
            modellist_imgs[n] = load_model_meanIMG_asnp(n)
        print('rectified imgs of indx: {} are loaded'.format(n))
    return modellist_imgs

#----------------------------------------------------------------------
def load_train_val_VGGOccupancy_VGGFeature_asnp(train_set, val_set, models_densityCube_param):
    """
    to load the VGG_triplet_Net occupancy and the feature vector of patches in different views of perticular nth model, 
    return: similiar with the output of the func 'get_VGG_triplet_Net_featureVec'
    feature_embed: np{N_pts, N_views, feature_embed_dim}
    desicion_onSurf: np{N_pts,} with 0/1 to 
    """
    train_val_models_VGGOccupancy_list = [[],[]]
    train_val_models_VGGFeature_list = [[],[]]
    for setIndx, modelIndx_list in enumerate([train_set, val_set]):
        if modelIndx_list == []:
            continue
        for n in modelIndx_list:
            model_VGGNet_npy_file = voxelVolume_txt_fld+'VGG_triplet_Net_outputs/{}.npz'.format(n)
            if os.path.exists(model_VGGNet_npy_file): 
                loaded_data = np.load(model_VGGNet_npy_file)
                train_val_models_VGGOccupancy_list[setIndx].append(loaded_data['arr_0'])
                train_val_models_VGGFeature_list[setIndx].append(loaded_data['arr_1'])
            else:
                tmp_VGGOccupancy, tmp_VGGFeature = load_model_VGGOccupancy_VGGFeature_asnp(n, models_densityCube_param[n])
                train_val_models_VGGOccupancy_list[setIndx].append(tmp_VGGOccupancy)
                train_val_models_VGGFeature_list[setIndx].append(tmp_VGGFeature)                
            print('rectified VGGNet of indx: {} are loaded'.format(n))
        VGGFeature_shape = train_val_models_VGGFeature_list[setIndx][0].shape
    ##VGGFeature_shape = train_val_models_VGGFeature_list[0][0].shape
            
    return np.asarray(train_val_models_VGGOccupancy_list[0], dtype=np.uint8).flatten(), \
           np.asarray(train_val_models_VGGFeature_list[0], dtype=np.float32).reshape((-1, VGGFeature_shape[1], VGGFeature_shape[2])),\
           np.asarray(train_val_models_VGGOccupancy_list[1], dtype=np.uint8).flatten(), \
           np.asarray(train_val_models_VGGFeature_list[1], dtype=np.float32).reshape((-1, VGGFeature_shape[1], VGGFeature_shape[2]))           


#----------------------------------------------------------------------
def load_modellist_densityCube_aslist(modelIndx_list, read_normal_infor_ON = False):
    """
    load the density cubes' txt representing the surface and generated by pcl
    return two lists, the param list and the data list
    read_normal_infor_ON: when == ON, the normal xyz information will be read from the saved txt file
    """
    modellist_densityCubes_param = [None for _ in range(max(modelIndx_list)+1)] ## only the position of indexes in the modelIndx_list are non-None
    modellist_densityCubes = [None for _ in range(max(modelIndx_list)+1)] ## only the position of indexes in the modelIndx_list are non-None
    modellist_CubesNormalxyz = [None for _ in range(max(modelIndx_list)+1)]

    for _modelIndx in modelIndx_list:
        densityCubes_fnm = voxelVolume_txt_fld+'output_stl_{:03}.txt'.format(_modelIndx)
        with open(densityCubes_fnm,mode='r') as f:
            lines = f.readlines()
            cubes_param = np.zeros((len(lines), N_cubeParams), dtype=np.float64)
            densityCube_s = np.zeros((len(lines), D, D, D), dtype=np.uint8)
            CubeNormalxyz_s, CubeNormalxyz = None, None
            if read_normal_infor_ON:
                CubeNormalxyz_s = np.zeros((len(lines), 3, D, D, D), dtype=np.float32)
            for _i, l in enumerate(lines):
                ## including min_x/y/z & resolution & modelIndx
                cube_param = np.asarray(l.split(',')[0].split(' '),dtype=np.float64)
                cubes_param[_i] = np.append(cube_param,_modelIndx)
                ## convert each line into an array, whose 4D row presents a 3D voxel's sparse density
                sparsexyz = np.asarray([_l.split(' ') for _l in l.split(',')[1:-1] ], dtype=np.float32) # assume x/y/z/T will be all uint 
                densityCube = np.zeros((D,D,D), dtype=np.uint8)
                if read_normal_infor_ON:
                    CubeNormalxyz = np.zeros((3,D,D,D), dtype=np.float32)
                if sparsexyz.size != 0:
                    X, Y, Z, T = sparsexyz[:,0].astype(np.uint16), sparsexyz[:,1].astype(np.uint16), \
                        sparsexyz[:,2].astype(np.uint16), sparsexyz[:,3].astype(np.uint16)
                    densityCube[X,Y,Z] = T   
                    if read_normal_infor_ON:
                        NormalX, NormalY, NormalZ = sparsexyz[:,4], sparsexyz[:,5], sparsexyz[:,6]
                        CubeNormalxyz[:,X,Y,Z] = NormalX, NormalY, NormalZ  
                        
                densityCube_s[_i] = densityCube
                if read_normal_infor_ON:
                    CubeNormalxyz_s[_i] = CubeNormalxyz
            
            modellist_densityCubes_param[_modelIndx] = cubes_param
            modellist_densityCubes[_modelIndx] = densityCube_s
            if read_normal_infor_ON:
                modellist_CubesNormalxyz[_modelIndx] = CubeNormalxyz_s
                
            if visualize_when_generateData_ON:
                visualize_N_densities_pcl([densityCube_s[0],densityCube_s[1],densityCube_s[2]])
    print 'density Cube files are loaded'
    if read_normal_infor_ON:
        return modellist_densityCubes_param, modellist_densityCubes, modellist_CubesNormalxyz        
    else:
        return modellist_densityCubes_param, modellist_densityCubes           
           

#----------------------------------------------------------------------
def calcu_VGG_triplet_Net_weight(all_viewPair_dimIndx_np, feature_embed, similarity_thresh = VGG_triplet_thresh):
    """
    all_viewPair_dimIndx_np: store the view pair indxes, {N, NOofPairs, 2} or {N, 2}
    feature_embed: store the feature vector of the patches on different views
    similarity_thresh: smaller than this thresh means the two patches are similiar.
    return:
    similarity_01: indicate whether the patches are similar {N, NOofAllPairs}
    argsort_diff_X: such as: [[0,3,2,1],[2,1,0,3]], 0 is the position of the max value along axis=-1
    """
    if all_viewPair_dimIndx_np.ndim == 3:
        _N, _NOofPairs, _ = all_viewPair_dimIndx_np.shape
        indx_h,indx_w=np.meshgrid(range(0,_N),range(0,_NOofPairs),indexing='ij')
        all_viewPair_dimIndx_np = all_viewPair_dimIndx_np.reshape((-1,2))
    else:
        _N, _NOofPairs = feature_embed.shape[0], all_viewPair_dimIndx_np.shape[0]
        indx_h,indx_w=np.meshgrid(range(0,_N),range(0,_NOofPairs),indexing='ij')
        
    diff_X = feature_embed[indx_h.flatten(),list(all_viewPair_dimIndx_np[:,0])*_N,:] - feature_embed[indx_h.flatten(),list(all_viewPair_dimIndx_np[:,1])*_N,:] 
    diff_X_sq_sum = (diff_X**2).sum(axis=-1)
    diff_X_Euclid = diff_X_sq_sum ** .5         
    similarity_01 = diff_X_Euclid <= similarity_thresh[1]
    diff_X_Euclid[diff_X_Euclid>similarity_thresh[1]] = 0
    argsort_diff_X = np.argsort( - diff_X_Euclid.reshape((_N,_NOofPairs)), axis=-1) ## firstcolumn is the index of the largest element
    return similarity_01.reshape((_N,_NOofPairs)), argsort_diff_X

            
            
            
def RECONSTR_generate_3DSamples_param_np(modelIndx):
    """
    func for RECONSTR
    generate {N_cubes} 3D non-overlapped cubes, each one has {N_cubeParams} dim
    for the cube with size of DxDxD, the valid prediction region is the center part, say, dxdxd
    The default grids of the cube is 32x32x32, i.e. D=32, d can be = 20.
    the param part: check the defination of the 'N_cubeParams'    
    return: 
    Samples_param_np, np{N_cubes, N_cubeParams}
    cube_Dsize, scalar
    """
    resol = np.float32(0.4) # resolution / the distance between adjacent grids
    cube_Dsize = resol * D_randcrop   # D size of each cube, 
    cube_Center_Dsize = resol * D_center   # D size of each cube's center, 
    cube_stride = cube_Center_Dsize #/ 2  # the distance between adjacent cubes, 
    ## for the +-300;+-300;400/1000 pair, the first 7W points contain almost nothing.
    x_=np.arange(-300,300,cube_stride).astype(np.float32) #+-300 
    y_=np.arange(-300,300,cube_stride).astype(np.float32) #+-300 
    z_=np.arange(400,1000,cube_stride).astype(np.float32) #400,1000    / 600, 800
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij') ## so that the x,y,z will by aligned with the sample_xyz_indices = np.indices(..)
    N_cubes = x.size
    Samples_param_np = np.zeros((N_cubes,N_cubeParams))
    Samples_param_np[:,0] = x.flatten()
    Samples_param_np[:,1] = y.flatten()
    Samples_param_np[:,2] = z.flatten()
    Samples_param_np[:,3] = resol
    Samples_param_np[:,4] = int(modelIndx)
    # store the xyz_indices of each sample, in order to localize the cube
    sample_xyz_indices = np.indices((x_.size,y_.size,z_.size)) 
    Samples_param_np[:,5] = sample_xyz_indices[0].flatten()
    Samples_param_np[:,6] = sample_xyz_indices[1].flatten()
    Samples_param_np[:,7] = sample_xyz_indices[2].flatten()
    
    return Samples_param_np, cube_Dsize
    
           
def RECONSTR_select_valid_pts_in_cube(predict, rgb_3D, param, VGGOccupancy):
    """
    only reserve the center part of the cube, because the boarder of our prediction is not accurate.
    
    predict: {N,1,D,D,D}
    rgb_3D: {N,3,D,D,D}
    param: {N,N_cubeParams}
    VGGOccupancy: {N,}
    return: xyz_rgba_np, {N_pts, 4}
    """
    valid_MinMax_indx = [(D_randcrop-D_center)/2, (D_randcrop-D_center)/2 + D_center - 1]
    predict[~(VGGOccupancy.astype(np.bool))] = 0
    valid_pts_indx = np.argwhere(predict>=1) 
    center_pts_indx = ((valid_pts_indx >= valid_MinMax_indx[0]) & (valid_pts_indx <= valid_MinMax_indx[1]))[:,3:].all(axis=1) # make sure all the xyz=n_xyz[-3:] are inscope
    valid_pts_indx = valid_pts_indx[center_pts_indx]
    N_pts = valid_pts_indx.shape[0]
    RGB_uint8 = ((rgb_3D+.5)*255).astype(np.uint8)
    A = np.asarray(254, dtype=np.uint8)        
    xyz_rgba_np = np.zeros((N_pts, 4))
    for i in xrange(N_pts):
        # valid_pts_indx: {N_pts, predict.ndim}
        n, _, ix, iy, iz = valid_pts_indx[i]
        x_min, y_min, z_min, resol, modelIndx = param[n][:5]
        x, y, z = x_min+ix*resol, y_min+iy*resol, z_min+iz*resol
        RGB = RGB_uint8[n,:,ix,iy,iz]
        ## thanks to: 'http://www.pcl-users.org/How-to-convert-from-RGB-to-float-td4022278.html'
        rgba = struct.unpack('f', chr(RGB[2]) + chr(RGB[1]) + chr(RGB[0]) + chr(A))[0]        
        xyz_rgba_np[i] = x, y, z, rgba
    
    return xyz_rgba_np

def RECONSTR_save_pcd(xyzrgba_stack, model_folder, filename):
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


def threshold(xlist, min, max):
    output = []
    for x in xlist:
        x[x<min] = min
        x[x>max] = max
        output.append(x)
    return output


def save_tmp_visual_file(x,y,z,R,G,B,density,normalX=None, normalY=None, normalZ=None):
    tmp_file_nm = '{:1.8f}.tmp'.format(time.time())
    if not (x.size==y.size==z.size==density.size):
        print 'error: size of x/y/z/density should be the same'
        return None
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)    
    with open(temp_folder+tmp_file_nm,'w') as f:
        for n in xrange(x.size):
            if (normalX is not None) and (normalY is not None) and (normalZ is not None):
                f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(x[n],y[n],z[n],R[n],G[n],B[n],density[n],normalX[n],normalY[n],normalZ[n]))
            else:
                f.write('{} {} {} {} {} {} {}\n'.format(x[n],y[n],z[n],R[n],G[n],B[n],density[n]))
    return temp_folder+tmp_file_nm+' '
        
        
def visualize_N_densities_pcl(coloredCube_list, normalCube_list=None, density_list=None):
    N_subplt = len(coloredCube_list)
    files_2_visual=' '
    os.system('rm '+temp_folder+'*') # remember to clear the tmp_file_folder    
    for _n, _coloredCube in enumerate(coloredCube_list):
        coloredCube = _coloredCube.astype(np.uint8)
        if coloredCube.ndim == 1:
            coloredCube = np.reshape(coloredCube, (D,D,D))
        if coloredCube.ndim >= 3:
            coloredCube = coloredCube.squeeze()
            if  coloredCube.ndim == 3: ## DxDxD, the value represents the density
                density = coloredCube
                density, = threshold([density], int(pts_in_voxel_MIN), int(pts_in_voxel_MAX))
                density = density.round() # for visualization, small values like 0.1 can also be shown
                X, Y, Z = density.nonzero()
                T = density[X,Y,Z]       
                R, G, B = T*10, 100+T*10, T*10
                if normalCube_list is not None:
                    if normalCube_list[_n] is not None:
                        normalCube = normalCube_list[_n].squeeze()
                        nX, nY, nZ = normalCube[0,X,Y,Z],normalCube[1,X,Y,Z],normalCube[2,X,Y,Z]
                        nX, nY, nZ = nX.flatten(), nY.flatten(), nZ.flatten()
                        files_2_visual += save_tmp_visual_file(X,Y,Z,R,G,B,T.astype(np.uint8),nX, nY, nZ)
                        continue                
            elif coloredCube.ndim == 4: ## ChannelxDxDxD, the value represent the RGB values, in order to visualize, let density = 1
                ##meshgrid: indexing : {'xy', 'ij'}, optional     ##Cartesian ('xy', default) or matrix ('ij') indexing of output.    
                #_D = coloredCube.shape[-1]
                #X,Y,Z = np.meshgrid(range(0,_D),range(0,_D),range(0,_D),indexing='ij')                  
                #R,G,B = coloredCube[0],coloredCube[1],coloredCube[2]
                if density_list is not None:
                    if density_list[_n] is not None:
                        T = density_list[_n].astype(np.uint8)
                    else:
                        T = coloredCube.any(axis=0) #Bug: the points whose RGB=(0,0,0) will be ignored.
                X,Y,Z = T.nonzero() # Now, it only show the nonZero parts. 
                R,G,B = coloredCube[0,X,Y,Z],coloredCube[1,X,Y,Z],coloredCube[2,X,Y,Z]
                T = np.ones(R.shape, dtype=np.uint8)
                X,Y,Z,R,G,B,T = X.flatten(),Y.flatten(),Z.flatten(),R.flatten(),G.flatten(),B.flatten(),T.flatten()
                
            else:   
                print 'error: the visualized array\'s shape != 1 or 3'

        files_2_visual += save_tmp_visual_file(X,Y,Z,R,G,B,T.astype(np.uint8))
    
    os.system(visualizer+files_2_visual) 
    print 'visualizer is waiting for enter to continue...'
    raw_input() # wait for 'enter'  


def load_1Ddata():
   
    filenm_train_data_noisy = data_fld+'train_data_noisy.npy'
    filenm_train_data_gt = data_fld+'train_data_gt.npy'
    filenm_val_data_noisy = data_fld+'val_data_noisy.npy'
    filenm_val_data_gt = data_fld+'val_data_gt.npy'
    if os.path.exists(filenm_train_data_noisy) and \
       os.path.exists(filenm_train_data_gt) and \
       os.path.exists(filenm_val_data_noisy ) and \
       os.path.exists(filenm_val_data_gt) :
        print('train val data files exist')
        train_data_noisy = np.load(filenm_train_data_noisy)
        train_data_gt = np.load(filenm_train_data_gt)
        val_data_noisy = np.load(filenm_val_data_noisy)
        val_data_gt = np.load(filenm_val_data_gt)
            
    else:
        print('train val data files don\'t exist')
        train_data_noisy, train_data_gt = read_setof_model_file(train_set)
        #visualize_2_densities(train_data_gt[12],train_data_noisy[12])
        val_data_noisy, val_data_gt = read_setof_model_file(val_set)
        train_data_noisy, train_data_gt, val_data_noisy, val_data_gt = threshold(\
            [train_data_noisy, train_data_gt, val_data_noisy, val_data_gt], int(pts_in_voxel_MIN), int(pts_in_voxel_MAX))
        with open(filenm_train_data_noisy,'w') as f:
            np.save(f,train_data_noisy.astype(np.uint8) )
        with open(filenm_train_data_gt,'w') as f:
            np.save(f,train_data_gt.astype(np.uint8) )
        with open(filenm_val_data_noisy,'w') as f:
            np.save(f,val_data_noisy.astype(np.uint8) )
        with open(filenm_val_data_gt,'w') as f:
            np.save(f, val_data_gt.astype(np.uint8))            
            
    
    return train_data_noisy.astype(np.float32)/pts_in_voxel_MAX, \
           train_data_gt.astype(np.float32)/pts_in_voxel_MAX, \
           val_data_noisy.astype(np.float32)/pts_in_voxel_MAX, \
           val_data_gt.astype(np.float32)/pts_in_voxel_MAX

def save_sparse_csr(filename,array):
    array_sparse = scipy.sparse.csr_matrix(array)
    np.savez(filename,data = array_sparse.data ,indices=array_sparse.indices,
             indptr =array_sparse.indptr, shape=array_sparse.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape']).toarray()

def load_1Ddata_sparse():
    filenm_train_data_noisy = data_fld+'train_data_noisy.npy'
    filenm_train_data_gt = data_fld+'train_data_gt.npy'
    filenm_val_data_noisy = data_fld+'val_data_noisy.npy'
    filenm_val_data_gt = data_fld+'val_data_gt.npy'
    if os.path.exists(filenm_train_data_noisy) and \
       os.path.exists(filenm_train_data_gt) and \
       os.path.exists(filenm_val_data_noisy ) and \
       os.path.exists(filenm_val_data_gt) :
        print('train val data files exist')
        train_data_noisy = load_sparse_csr(filenm_train_data_noisy)
        train_data_gt = load_sparse_csr(filenm_train_data_gt)
        val_data_noisy = load_sparse_csr(filenm_val_data_noisy)
        val_data_gt = load_sparse_csr(filenm_val_data_gt)
            
    else:
        print('train val data files don\'t exist')
        train_data_noisy, train_data_gt = read_setof_model_file(train_set)
        #visualize_2_densities(train_data_gt[12],train_data_noisy[12])
        val_data_noisy, val_data_gt = read_setof_model_file(val_set)
        train_data_noisy, train_data_gt, val_data_noisy, val_data_gt = threshold(\
            [train_data_noisy, train_data_gt, val_data_noisy, val_data_gt], int(pts_in_voxel_MIN), int(pts_in_voxel_MAX))
        with open(filenm_train_data_noisy,'w') as f:
            save_sparse_csr(f,train_data_noisy )
        with open(filenm_train_data_gt,'w') as f:
            save_sparse_csr(f,train_data_gt)
        with open(filenm_val_data_noisy,'w') as f:
            save_sparse_csr(f,val_data_noisy )
        with open(filenm_val_data_gt,'w') as f:
            save_sparse_csr(f, val_data_gt) ## don't save as int, otherwise when reload the data, all the dtype will be int.           
            
    
    return train_data_noisy.astype(np.float32)/pts_in_voxel_MAX, \
           train_data_gt.astype(np.float32)/pts_in_voxel_MAX, \
           val_data_noisy.astype(np.float32)/pts_in_voxel_MAX, \
           val_data_gt.astype(np.float32)/pts_in_voxel_MAX


def load_3Ddata(LowResolution = False):
    train_noisy, train_gt, val_noisy, val_gt = load_1Ddata_sparse()
    ##N_voxels = train_gt.shape[-1]
    ##d = int(N_voxels ** (1/3.) + .5) # round, not elegant at all!
    if LowResolution:
        for dataset_stack in [train_noisy, val_noisy]:
            for dataset in dataset_stack:
                randIndx = np.asarray(range(N))
                np.random.shuffle(randIndx)
                dataset[randIndx[:15*N/16]] = 0       
    return [x.reshape((x.shape[0],1)+(D,)*3) for x in [train_noisy, train_gt, val_noisy, val_gt]]
    

#----------------------------------------------------------------------
def map_pixHW_2_uintIDnp(pt_w,pt_h):
    """ 
    used in func: 'colorize_cube' 
    np([21,20,22,21,21]),np([10,11,12,13,13]) ==> np([0,1,2,3,3])
    """
    ##range_w = pt_w.max()-pt_w.min()+1
    ##pixID_value = (pt_h-pt_h.min()) * range_w + (pt_w-pt_w.min())
    ##unique_pixID_value = set(pixID_value)
    ##mapping = {_value:_indx for _indx, _value in enumerate(unique_pixID_value)}
    ##return np.asarray(map(mapping.get, pixID_value),dtype=np.uint16)
    # above version: max_indx = 20K, 
    # below refined version: max_indx = 13K
    ##wh_tupleList = [(w,h) for w, h in zip(pt_w, pt_h)]
    ##wh_tupleList_set=set(wh_tupleList)
    ##mapping = {_wh_tuple:_wh_tuple_indx for _wh_tuple_indx, _wh_tuple in enumerate(wh_tupleList_set)}
    ##return np.asarray(map(mapping.get, wh_tupleList),dtype=np.uint16)
    # above version: because of using list, takes long time
    # below version: only use numpy to implement. Thanks: http://stackoverflow.com/questions/29535341/generate-unique-values-based-on-rows-in-a-numpy-array
    pt_wh_np = np.asarray([pt_w,pt_h]).T
    pt_wh_sort_indx = np.lexsort(pt_wh_np.T)
    pt_wh_sort_diff = np.diff(pt_wh_np[pt_wh_sort_indx], axis=0)
    maskID_sort = np.cumsum(np.append([True],pt_wh_sort_diff.any(axis=1)))
    maskID = np.zeros(pt_wh_sort_indx.shape, dtype=np.uint16)
    maskID[pt_wh_sort_indx] = maskID_sort   
    return maskID

#----------------------------------------------------------------------
def colorize_cube(a_densityCube_param_data, view_set, cameraPOs_np, model_imgs_np, visualization_ON=False, return_pixIDmask_cubes_ON = False):
    """ 
    generate colored cubes of a perticular densityCube  
    inputs: a_densityCube_param_data=[cube_param_np, cube_data_np]
    output: 
    [views_N, 3, D, D, D]. 3 is for RGB
    pixIDmask_cubes: 
    """
    N_views = len(view_set)
    max_h, max_w, _ = model_imgs_np[0].shape
    min_x,min_y,min_z,resol,modelIndx = a_densityCube_param_data[0][:5]
    densityCube = a_densityCube_param_data[1]
    indx_xyz = range(0,D)
    ##meshgrid: indexing : {'xy', 'ij'}, optional     ##Cartesian ('xy', default) or matrix ('ij') indexing of output.    
    indx_x,indx_y,indx_z = np.meshgrid(indx_xyz,indx_xyz,indx_xyz,indexing='ij')  
    indx_x = indx_x * resol + min_x
    indx_y = indx_y * resol + min_y
    indx_z = indx_z * resol + min_z
    homogen_1s = np.ones(D**3, dtype=np.float64)
    pts_4D = np.vstack([indx_x.flatten(),indx_y.flatten(),indx_z.flatten(),homogen_1s])
    
    colored_cubes = np.zeros((N_views,3,D,D,D))
    pixIDmask_cubes = np.zeros((N_views,1,D,D,D), dtype=np.uint16) if return_pixIDmask_cubes_ON else None
    
    # only chooce from inScope views
    ##center_pt_xyz1 = np.asarray([D*resol/2 + min_x, D*resol/2 + min_y, D*resol/2 + min_z, 1])
    ##center_pt_3D = np.dot(cameraPOs_np,center_pt_xyz1)
    ##center_pt_wh = center_pt_3D[:,:-1] / center_pt_3D[:,-1:]# the result is vector: [w,h,1], w is the first dim!!!
    ##valid_views = (center_pt_wh[:,0]<max_w) & (center_pt_wh[:,1]<max_h) & (center_pt_wh[:,0]>0) & (center_pt_wh[:,1]>0)      
    ##while valid_views.sum() < N_views: ## if only n views can see this pt, where n is smaller than N_views, randomly choose some more
        ##valid_views[random.randint(1,cameraPOs_np.shape[0]-1)] = True
    ##valid_view_list = list(valid_views.nonzero()[0]) ## because the cameraPOs_np[0] is zero, don't need +1 here
    ##view_list = random.sample(valid_view_list,N_views)    
    
    for _n, _view in enumerate(view_set):
        # perspective projection
        projection_M = cameraPOs_np[_n] ## use dimIndx
        pts_3D = np.dot(projection_M, pts_4D)
        pts_3D[:-1] /= pts_3D[-1] # the result is vector: [w,h,1], w is the first dim!!!
        pts_2D = pts_3D[:-1].round().astype(np.uint16)
        pts_w, pts_h = pts_2D[0], pts_2D[1]
        # access rgb of corresponding model_img using pts_2D coordinates
        pts_RGB = np.zeros((D**3, 3))
        img = model_imgs_np[_view] ## use viewIndx
        inScope_pts_indx = (pts_w<max_w) & (pts_h<max_h) & (pts_w>0) & (pts_h>0)
        pts_RGB[inScope_pts_indx] = img[pts_h[inScope_pts_indx],pts_w[inScope_pts_indx]]
        colored_cubes[_n] = pts_RGB.T.reshape((3,D,D,D))
        if return_pixIDmask_cubes_ON:
            pts_pixID = map_pixHW_2_uintIDnp(pts_w, pts_h)
            pixIDmask_cubes[_n] = pts_pixID.reshape((1,D,D,D))
        
    if visualization_ON:    
        visualize_N_densities_pcl([densityCube]+[colored_cubes[n] for n in range(0,len(5))])
        
    if return_pixIDmask_cubes_ON:
        return colored_cubes, pixIDmask_cubes
    else:
        return colored_cubes
    

#----------------------------------------------------------------------            
def gen_coloredCubes(view_set, N_viewPairs, occupiedCubes_param, cameraPOs, models_img, diff_X_Euclid, \
                     visualization_ON = False, occupiedCubes_01=None, similarity_thresh = VGG_triplet_thresh[1],\
                     return_pixIDmask_cubes_ON=False):     
    """
    inputs: 
    view_set: where the view pairs are chosen from
    N_viewPairs: scalar, represents how many view pairs 
    occupiedCubes_param: parameters for each occupiedCubes (N,params)
    occupiedCubes_01: multiple occupiedCubes (N,D,D,D)
    return:
    coloredCubes = (N*N_viewPairs,3*2,D,D,D) 
    selected_viewPair_dimIndx_np: {N,N_viewPairs,2}, make sure the patches from each view pair are similiar(use the func'calcu_VGG_triplet_Net_weight' to determine).
    VGG_triplet_Net_weight_01: (N, N_viewPairs)
    """
    N_views = len(view_set)
    N_cubes = occupiedCubes_param.shape[0]
    coloredCubes = np.zeros((N_cubes,N_viewPairs*2,3,D,D,D), dtype=np.float32) # reshape at the end
    if return_pixIDmask_cubes_ON:
        pixIDmask_cubes = np.zeros((N_cubes,N_viewPairs*2,1,D,D,D), dtype=np.uint16) # reshape at the end
         
    selected_viewPair_dimIndx_np = np.zeros((N_cubes, N_viewPairs, 2), dtype=np.uint16)
    
    ##VGG_weight_01_allViewPairs, VGG_weight_argsort_allViewPairs = calcu_VGG_triplet_Net_weight(all_viewPair_dimIndx_np, feature_embed, \
                                                                                               ##similarity_thresh = similarity_thresh)
    VGG_weight_01_allViewPairs = (diff_X_Euclid < similarity_thresh) & (diff_X_Euclid > .1)
    VGG_weight_argsort_allViewPairs = np.argsort( - diff_X_Euclid, axis=-1)
    
    VGG_weight_01_selectViewPairs = np.zeros((N_cubes, N_viewPairs), dtype=np.uint8)
    for _n_cube in range(0, N_cubes): ## each cube
        timer_start = time.time()
        occupiedCube_param = occupiedCubes_param[_n_cube]
        if visualization_ON:
            if occupiedCubes_01 is None:
                print 'error: [func]gen_coloredCubes, occupiedCubes_01 should not be None when visualization_ON==True'
            occupiedCube_01 = occupiedCubes_01[_n_cube]
        else:
            occupiedCube_01 = None
        ##randViewIndx = random.sample(range(1,cameraPOs.shape[0]),N_views)
            
        model_indx = int(occupiedCube_param[4])

        occupiedCube_param_01 = [occupiedCube_param, occupiedCube_01]
        coloredCube = colorize_cube(a_densityCube_param_data = occupiedCube_param_01, view_set=view_set, \
                      cameraPOs_np = cameraPOs, model_imgs_np = models_img[model_indx], visualization_ON=visualization_ON,\
                      return_pixIDmask_cubes_ON=return_pixIDmask_cubes_ON)
        if return_pixIDmask_cubes_ON:
            coloredCube, pixIDmask_cube = coloredCube ## there are 2 outputs
         
        #select the view pairs with largest VGG_disimilarity value, because the view pairs will be far away and be good for surface predication.
        valid_pairIndx_Indx = VGG_weight_argsort_allViewPairs[_n_cube] 
        ##valid_pairIndx_Indx, = np.where(VGG_weight_01_allViewPairs[_n_cube]==1)
        ##if valid_pairIndx_Indx.size < N_viewPairs:
            ##valid_pairIndx_Indx = np.hstack([valid_pairIndx_Indx,random.sample(range(0,all_pairIndx_np.shape[0]), N_viewPairs)])
        VGG_weight_01_selectViewPairs[_n_cube] = VGG_weight_01_allViewPairs[_n_cube,valid_pairIndx_Indx[:N_viewPairs]]
        
        ##select_pairIndx = [all_pairIndx[i] for i in valid_pairIndx_Indx[:N_viewPairs]]
        ##selected_viewPair_dimIndx_sequence = [x for pair_tuple in select_pairIndx for x in pair_tuple] ## [(a,),(a,b),(a,b,c)] ==> [a,a,b,a,b,c]
        ##selected_viewPair_dimIndx_np[_n_cube] = np.asarray(selected_viewPair_dimIndx_sequence).reshape((-1,2))
        selected_viewPair_dimIndx_np[_n_cube] = all_viewPair_dimIndx_np[valid_pairIndx_Indx[:N_viewPairs],:]
        selected_viewPair_dimIndx_sequence = all_viewPair_dimIndx_np[valid_pairIndx_Indx[:N_viewPairs],:].flatten()
        
        coloredCubes[_n_cube] = coloredCube[selected_viewPair_dimIndx_sequence]
        if return_pixIDmask_cubes_ON:
            pixIDmask_cubes[_n_cube] = pixIDmask_cube[selected_viewPair_dimIndx_sequence]
            
    if return_pixIDmask_cubes_ON:  
        return coloredCubes.reshape((N_cubes*N_viewPairs,3*2,D,D,D)), selected_viewPair_dimIndx_np, VGG_weight_01_selectViewPairs,\
               pixIDmask_cubes.reshape((N_cubes*N_viewPairs,1*2,D,D,D))
    else: 
        return coloredCubes.reshape((N_cubes*N_viewPairs,3*2,D,D,D)), selected_viewPair_dimIndx_np, VGG_weight_01_selectViewPairs


#----------------------------------------------------------------------            
def gen_coloredCubes_withoutVGGTriplet(N_views, N_viewPairs, occupiedCubes_param, cameraPOs, models_img, \
                                       visualization_ON = False, occupiedCubes_01=None,return_pixIDmask_cubes_ON=False):     
    """
    copied from the version:  20160520-01-6.e
    inputs: 
    N_views: where the view pairs are chosen from
    N_viewPairs: scalar, represents how many view pairs 
    occupiedCubes_param: parameters for each occupiedCubes (N,params)
    occupiedCubes_01: multiple occupiedCubes (N,D,D,D)
    return:
    coloredCubes = (N*N_viewPairs,3*2,D,D,D) 
    pixIDmask_cubes = (N*N_viewPairs,1*2,D,D,D), uint16
    """
    N_cubes = occupiedCubes_param.shape[0]
    coloredCubes = np.zeros((N_cubes,N_viewPairs*2,3,D,D,D), dtype=np.float32) # reshape at the end
    if return_pixIDmask_cubes_ON:
        pixIDmask_cubes = np.zeros((N_cubes,N_viewPairs*2,1,D,D,D), dtype=np.uint16) # reshape at the end
    
    for _n_cube in range(0, N_cubes): ## each cube
        timer_start = time.time()
        occupiedCube_param = occupiedCubes_param[_n_cube]
        if visualization_ON:
            if occupiedCubes_01 is None:
                print 'error: [func]gen_coloredCubes, occupiedCubes_01 should not be None when visualization_ON==True'
            occupiedCube_01 = occupiedCubes_01[_n_cube]
        else:
            occupiedCube_01 = None
        ##randViewIndx = random.sample(range(1,cameraPOs.shape[0]),N_views)
            
        model_indx = int(occupiedCube_param[4])

        occupiedCube_param_01 = [occupiedCube_param, occupiedCube_01]
        coloredCube = colorize_cube(a_densityCube_param_data = occupiedCube_param_01, N_views=N_views, \
                      cameraPOs_np = cameraPOs, model_imgs_np = models_img[model_indx], visualization_ON=visualization_ON, \
                      return_pixIDmask_cubes_ON=return_pixIDmask_cubes_ON)
        if return_pixIDmask_cubes_ON:
            coloredCube, pixIDmask_cube = coloredCube ## there are 2 outputs
            
        # [a,b,c] ==> [a,b,a,c,b,c]
        ##all_pairIndx = ()
        ##for _pairIndx in itertools.combinations(range(0,N_views),2):
            ##all_pairIndx += _pairIndx
        ##all_pairIndx = list(all_pairIndx)
        
        # [a,b,c,d,e,f,g,h,i,j] ==> [a,b,g,c,f,e]
        all_pairIndx = []
        for _pairIndx in itertools.combinations(range(0,N_views),2):
            all_pairIndx.append(_pairIndx)
        all_pairIndx = random.sample(all_pairIndx, N_viewPairs)
        all_pairIndx = [x for pair_tuple in all_pairIndx for x in pair_tuple] ## [(a,),(a,b),(a,b,c)] ==> [a,a,b,a,b,c]
        
        coloredCubes[_n_cube] = coloredCube[all_pairIndx]
        if return_pixIDmask_cubes_ON:
            pixIDmask_cubes[_n_cube] = pixIDmask_cube[all_pairIndx]
        
    if return_pixIDmask_cubes_ON:  
        return coloredCubes.reshape((N_cubes*N_viewPairs,3*2,D,D,D)), \
               pixIDmask_cubes.reshape((N_cubes*N_viewPairs,1*2,D,D,D))
    else: 
        return coloredCubes.reshape((N_cubes*N_viewPairs,3*2,D,D,D))


#----------------------------------------------------------------------            
def inScope_check(a_densityCube_param_data, N_inScopeViews, cameraPOs_np, model_imgs_np):
    """ 
    check how many views can see this cube. If the NO is smaller than N_inScopeViews, return None.
    inputs: a_densityCube_param_data=[cube_param_np, cube_data_np]
    """
    max_h, max_w, _ = model_imgs_np[0].shape
    min_x,min_y,min_z,resol,modelIndx = a_densityCube_param_data[0][:5]
        
    # only chooce from inScope views
    center_pt_xyz1 = np.asarray([D*resol/2 + min_x, D*resol/2 + min_y, D*resol/2 + min_z, 1])
    center_pt_3D = np.dot(cameraPOs_np,center_pt_xyz1)
    center_pt_wh = center_pt_3D[:,:-1] / center_pt_3D[:,-1:]# the result is vector: [w,h,1], w is the first dim!!!
    valid_views = (center_pt_wh[:,0]<max_w) & (center_pt_wh[:,1]<max_h) & (center_pt_wh[:,0]>0) & (center_pt_wh[:,1]>0)      
    if valid_views.sum() < N_inScopeViews:
        return None
    return 0


#----------------------------------------------------------------------            
def inScope_Cubes(N_inScopeViews, occupiedCubes_param, occupiedCubes_01, cameraPOs, models_img):     
    """
    inputs: 
    N_inScopeViews: scalar, represents threshold of NO. for inScope views
    occupiedCubes_param: parameters for each occupiedCubes (N,params)
    occupiedCubes_01: multiple occupiedCubes (N,D,D,D)
    return:
    occupiedCubes_param_inScope: filtered
    occupiedCubes_01_inScope: filtered occupiedCubes (N,D,D,D)
    """
    N_cubes = occupiedCubes_param.shape[0]
    inScope_indices = np.ones((N_cubes), dtype=np.bool)
    for _n_cube in range(0, N_cubes): ## each cube
        occupiedCube_param = occupiedCubes_param[_n_cube]
        ##randViewIndx = random.sample(range(1,cameraPOs.shape[0]),N_views)
            
        model_indx = int(occupiedCube_param[4])

        occupiedCube_param_01 = [occupiedCube_param, None]
        inScope_check_ = inScope_check(a_densityCube_param_data = occupiedCube_param_01, N_inScopeViews=N_inScopeViews, \
                      cameraPOs_np = cameraPOs, model_imgs_np = models_img[model_indx])
        if inScope_check_ is None:
            inScope_indices[_n_cube] = False
            
    if occupiedCubes_01 is None:    
        return occupiedCubes_param[inScope_indices], None
    else:        
        return occupiedCubes_param[inScope_indices], occupiedCubes_01[inScope_indices]

    
#----------------------------------------------------------------------            
def inScope_nearSurf_Cubes(N_inScopeViews, occupiedCubes_param, VGGOccupancy, cameraPOs, models_img):     
    """
    The reason why the selected cubes are near to surf is because of the VGGOccupancy, acting as an initial filter.
    inputs:
    N_inScopeViews: scalar, represents threshold of NO. for inScope views
    occupiedCubes_param: parameters for each occupiedCubes (N,params)
    VGGOccupancy: VGG_triplet_Net's output for the occupancy of the cube(N,)
    return:
    occupiedCubes_param_inScope: filtered
    occupiedCubes_01_inScope: filtered occupiedCubes (N,D,D,D)
    """
    N_cubes = occupiedCubes_param.shape[0]
    inScope_indices = VGGOccupancy.astype(np.bool)
    for _n_cube in np.where(inScope_indices == True)[0]: ## each cube whose VGGOccupancy == True
        occupiedCube_param = occupiedCubes_param[_n_cube]
        ##randViewIndx = random.sample(range(1,cameraPOs.shape[0]),N_views)
            
        model_indx = int(occupiedCube_param[4])

        occupiedCube_param_01 = [occupiedCube_param, None]
        inScope_check_ = inScope_check(a_densityCube_param_data = occupiedCube_param_01, N_inScopeViews=N_inScopeViews, \
                      cameraPOs_np = cameraPOs, model_imgs_np = models_img[model_indx])
        if inScope_check_ is None:
            inScope_indices[_n_cube] = False
            
    return occupiedCubes_param[inScope_indices], inScope_indices
    

#----------------------------------------------------------------------            
def load_train_val_gt_asnp(visualization_ON=False):     
    """
    load train/val_gt, which include the param part and the data part
    the param part: check the defination of the 'N_cubeParams'
    the data part: 0/1 indicate whether this subcube can be considered as a surface
    the normal part: 0-->1 float 3D unit vector {N,3,D,D,D}
    """
    models_densityCube_param,models_densityCube, modellist_CubesNormalxyz = load_modellist_densityCube_aslist(train_set+val_set, read_normal_infor_ON=True)
    train_gt_param, train_gt_Density = [models_densityCube_param[i] for i in train_set],[models_densityCube[i] for i in train_set]
    val_gt_param, val_gt_Density = [models_densityCube_param[i] for i in val_set],[models_densityCube[i] for i in val_set]
    
    train_gt_Normal = [modellist_CubesNormalxyz[i] for i in train_set]
    val_gt_Normal = [modellist_CubesNormalxyz[i] for i in val_set]
    
    train_gt_param, val_gt_param = np.asarray(train_gt_param).reshape(-1,N_cubeParams), np.asarray(val_gt_param).reshape(-1,N_cubeParams)
    train_gt_01 = np.where(np.asarray(train_gt_Density).reshape(-1,D,D,D) > 2, 1, 0).astype(np.uint8) ## {N,D,D,D}
    val_gt_01 = np.where(np.asarray(val_gt_Density).reshape(-1,D,D,D) > 2, 1, 0).astype(np.uint8)
    
    train_gt_Normal = np.asarray(train_gt_Normal).reshape(-1,3,D,D,D) * train_gt_01[:,None,...] ## {N,3,D,D,D}
    val_gt_Normal = np.asarray(val_gt_Normal).reshape(-1,3,D,D,D) * val_gt_01[:,None,...]
    
    if visualization_ON:
        visualize_N_densities_pcl([train_gt_01[0],train_gt_01[100],val_gt_01[0],val_gt_01[100]])
    return train_gt_param, val_gt_param, train_gt_01, val_gt_01, train_gt_Normal, val_gt_Normal, models_densityCube_param
    
    

def data_augment_rand_rotate(X_list, X_2):
    ## do the changes to the two input arrays together
    ## do the augmentation on the ending 3 axises
    # 90k degree rotation augmentation: can be easily implemented by transpose + (::-1) operations
    ## randomly transpose the ending 3 axises, assume it's in 5D
    rand_Transpose = (0,1) + tuple(random.sample([2,3,4],3))
    X_list = [_x.transpose(rand_Transpose) for _x in X_list]
    X_2 = X_2.transpose(rand_Transpose)
    ## because the axises are randomly transposed, we can do the (::-1) operation at one axis only
    X_list = [_x[:,:,:,:,::-1] for _x in X_list]
    X_2 = X_2[:,:,:,:,::-1]
    return X_list, X_2


def data_augment_rand_rotate_withNormal(X_1, X_2, X_normal):
    ## do the changes to the two input arrays together
    ## do the augmentation on the ending 3 axises
    # 90k degree rotation augmentation: can be easily implemented by transpose + (::-1) operations
    ## randomly transpose the ending 3 axises, assume it's in 5D
    xyzdim_transposed_list = random.sample([2,3,4],3)
    rand_Transpose = (0,1) + tuple(xyzdim_transposed_list)
    X_1 = X_1.transpose(rand_Transpose)
    X_2 = X_2.transpose(rand_Transpose)
    X_normal = X_normal.transpose(rand_Transpose)
    ## when xyz transposed, for the normal vector maxtrix, we only need to correspondingly permute the xyz channel at the second dimension.
    X_normal = X_normal[:,np.asarray(xyzdim_transposed_list)-2,...] 
    ## because the axises are randomly transposed, we can do the (::-1) operation at one axis only
    X_1 = X_1[:,:,:,:,::-1]
    X_2 = X_2[:,:,:,:,::-1]
    X_normal = X_normal[:,:,:,:,::-1]
    ## when x/y/z dimension reflected, for the normal vector maxtrix, we only need to correspondingly let the x/y/z's channel's value *= -1.
    X_normal[:,-1] *= -1
    
    return X_1, X_2, X_normal


def data_augment_scipy_rand_rotate(X1,X2):
    """take year!!!"""
    theta1, theta2 = random.randint(0,360), random.randint(0,360)
    X1 = scipy.ndimage.interpolation.rotate(X1, theta1, axes=(-2,-1), reshape=False)
    X2 = scipy.ndimage.interpolation.rotate(X2, theta1, axes=(-2,-1), reshape=False)

    X1 = scipy.ndimage.interpolation.rotate(X1, theta2, axes=(-3,-2), reshape=False)
    X2 = scipy.ndimage.interpolation.rotate(X2, theta2, axes=(-3,-2), reshape=False)
    return X1.astype(np.float32), X2.astype(np.float32)



def data_augment_rand_crop(Xlist, crop_size = D_randcrop):
    # random crop on ending 3 dimensions of any tensor with D>=3
    randx,randy,randz = np.random.randint(0,D-crop_size+1,size=(3,))
    #[...,xxx], ... means 
    return [X[...,randx:randx+crop_size,randy:randy+crop_size,randz:randz+crop_size] for X in Xlist]
    
    
    
           
#----------------------------------------------------------------------            
# For test
##load_train_val_gt_asnp()
##gen_coloredCubes()
#if visualize_when_generateData_ON:
    #load_3Ddata(LowResolution=False)
##RECONSTR_generate_3DSamples_param_np(111)