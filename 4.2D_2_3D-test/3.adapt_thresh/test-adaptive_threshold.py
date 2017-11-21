import numpy as np
import os
import sys
import time
import math
import copy
import thread
from multiprocessing import Process
from train_val_data_adaptThresh import *
import scan_RayTracing as rayTrace
sys.path.append("../2.py-reconstruction")
import train_val_data_4Test
sys.path.append("../../1.VGG_triplet-train/2.vgg_prepare_py")
import prepare_data
from sklearn.decomposition import PCA
from utils_sparseDict import sparseDict
import utils_sparseDict
import rayPooling
from maxima import localmax_ndim, globmax_ndim
from plyfile import PlyData, PlyElement

def save_pcd_pcd2ply(points2save, pcl_pcd2ply_exe, pcd_folder, filename ):
    print("saving ply")
    RECONSTR_save_pcd(points2save, pcd_folder, filename)
    os.system( "{} {} {}".format(pcl_pcd2ply_exe, pcd_folder+filename,\
            (pcd_folder+filename).replace('.pcd','.ply')))
    

def refine_save_points_2_ply(refine_cubes_prediction_3Ddict, refine_adapThresh_np, refine_cubes_occupancy_np, refine_cubes_rgb_3Ddict, \
        refine_cubes_param_3Ddict, filename, pcd_folder, center_D_range, refine_cubes_normal_3Ddict, with_pt_normal = False):
    pcl_pcd2ply_exe = '~/Downloads/pcl-trunk/build/bin/pcl_pcd2ply'
    
    points_save = None
    _xyz = np.argwhere(refine_cubes_occupancy_np == True)       
    for _cube in xrange(_xyz.shape[0]):
        _x,_y,_z = _xyz[_cube]
        rgb_test = refine_cubes_rgb_3Ddict[_x, _y, _z][None,...]
        ##rgb_test = np.random.random((1,3,D,D,D)) - .5
        points_sub = train_val_data_4Test.RECONSTR_select_valid_pts_in_cube(refine_cubes_prediction_3Ddict[_x, _y, _z][None,...]*refine_adapThresh_np[_x,_y,_z]>=1, \
                                                                   rgb_test, refine_cubes_param_3Ddict[_x,_y,_z][None,...], np.asarray([1]), center_D_range = center_D_range,\
                                                                   pt_norm = refine_cubes_normal_3Ddict[_x,_y,_z][None,...] if with_pt_normal else None)
        points_save = points_sub if points_save is None else np.append(points_save, points_sub)
    
    if points_save is not None: 
        filename_ply = os.path.join(pcd_folder,filename+'.ply')
        el_vertex = PlyElement.describe(points_save, 'vertex')
        print("saving ply file: {}".format(filename_ply))
        try:
            p = Process(target=PlyData([el_vertex]).write, args=(filename_ply,))
            p.start()
            print 'saving ply in new process'
        except:
            PlyData([el_vertex]).write(filename_ply)
            print("Cannot start new process to saving ply. Run it in the orignal process")
            # save_pcd_pcd2ply(np.concatenate(points_save, axis=0), pcl_pcd2ply_exe, pcd_folder, filename)

    else:
        print 'no points left'



def access_side_2D(np_3D, neighbor_direction):
    ## neighbor_direction: 6 neighbors' direction of current 3D cube. select the cube's nearest surface to the neighbor
    if (neighbor_direction == [1,0,0]).all(): 
        return np_3D[-1,:,:]
    if (neighbor_direction == [0,1,0]).all(): 
        return np_3D[:,-1,:]
    if (neighbor_direction == [0,0,1]).all(): 
        return np_3D[:,:,-1]
    if (neighbor_direction == [-1,0,0]).all(): 
        return np_3D[0,:,:]
    if (neighbor_direction == [0,-1,0]).all(): 
        return np_3D[:,0,:]
    if (neighbor_direction == [0,0,-1]).all(): 
        return np_3D[:,:,0]

def access_half_cube_3D(np_3D, xyz_direction):
    ## select half cube according to the xyz_direction
    cube_size = np_3D.shape[-1]
    _mid_indx = cube_size/2
    if (xyz_direction == [1,0,0]).all(): 
        return np_3D[_mid_indx:,:,:]
    if (xyz_direction == [0,1,0]).all(): 
        return np_3D[:,_mid_indx:,:]
    if (xyz_direction == [0,0,1]).all(): 
        return np_3D[:,:,_mid_indx:]
    if (xyz_direction == [-1,0,0]).all(): 
        return np_3D[:_mid_indx,:,:]
    if (xyz_direction == [0,-1,0]).all(): 
        return np_3D[:,:_mid_indx,:]
    if (xyz_direction == [0,0,-1]).all(): 
        return np_3D[:,:,:_mid_indx]

def calc_normal_vector(cube_bool_np, return_variance=False):
    """
    calculate the normal vector of the 3D occupancy stored in numpy 
    """
    samples_3D = np.argwhere(cube_bool_np > 0)
    pca = PCA(n_components=3)
    pca.fit(samples_3D)
    if return_variance:
        return pca.components_[-1], pca.explained_variance_ # [-1] # eigenVector with the min eigenValue
    else:
        return pca.components_[-1] # eigenVector with the min eigenValue

#----------------------------------------------------------------------
def visualize_nxnxn_cubes(refine_cubes_surface_3Ddict, _x,_y,_z, Nx, Ny, Nz, nR):
    """
    in order to localize a specific _x_y_z cube
    use this func to visualize the 3x3x3 neighboring cubes of the cube: refine_cubes_surface_3Ddict[_x, _y, _z]
    nR: the visualized area: (2nR+1)*(2nR+1)*(2nR+1) = nD*nD*nD
    if nR = 0, only visualize the center cube
    """
    nD = 2*nR+1
    cube_shape = refine_cubes_surface_3Ddict[_x, _y, _z].shape
    display_array=np.zeros((nD,nD,nD)+cube_shape)
    cubes_indx=np.indices((nD,nD,nD)).reshape((3,-1)).T ## generate the array like [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2,0,0],[0,2,0],[0,0,2],...]
    for _cube_xyz in cubes_indx:
        _xyz_shift = _cube_xyz - nR
        _x_ovlp,_y_ovlp,_z_ovlp = np.asarray([_x,_y,_z])+_xyz_shift  
        if _x_ovlp>=Nx or _y_ovlp>=Ny or _z_ovlp>=Nz or refine_cubes_surface_3Ddict[_x_ovlp, _y_ovlp, _z_ovlp]==[]: 
            continue
        display_array[_cube_xyz[0],_cube_xyz[1],_cube_xyz[2]]=refine_cubes_surface_3Ddict[_x_ovlp, _y_ovlp, _z_ovlp]
    
    display_array_3D=np.transpose(display_array,(0,3,1,4,2,5))
    vis_tmp=display_array_3D.reshape((nD*cube_shape[0],nD*cube_shape[1],nD*cube_shape[2]))
    visualize_N_densities_pcl([vis_tmp])


def get_normalInfo_from_dict(normalInformation_3Ddict, key_xyz_xyzShift, halfCube=None):
    if not normalInformation_3Ddict.has_key(key_xyz_xyzShift):
        normalInformation_3Ddict[key_xyz_xyzShift] = calc_normal_vector(cube_bool_np = halfCube, return_variance=True)
    return normalInformation_3Ddict[key_xyz_xyzShift]


enable_maskID_pool = True
visualization_ON = False # debug_ON
with_pt_normal = False # doubled the memory assumption!!!
# read_maskID_pooled_data_ON = False
debug_ON = False  
RGB_tmp_results = True # debug_ON
check_rayPooling = False # debug_ON
load_origPrediction = visualization_ON or enable_maskID_pool
print("load prediction: " + ("orig" if load_origPrediction else "globMax"))

old_adpthresh_policy = False

N_viewPairs = 1 # 2/6
thresh_NO_viewPairs_decisions_list = [10/12.0*N_viewPairs*2] #[2, 1] #[12,2,5,8,16] if enable_maskID_pool else [0]  #[0,2,4,6,8,10,12,14,16]

init_adapThresh_list = [2.0] # if not debug_ON else [2.0] #[1.5]
min_adapThresh = 1.2 #init_adapThresh_list[0] - .5
max_adapThresh = init_adapThresh_list[0] + .0 # in order to save memory, only load the cubes with max prediction > 1/3.0
N_refine_iter = 16
desired_thickness = 1.0

modelIndx = 17 #3/68; 12/91/[0,92]; 17/75/[0,76];
N_batch_files = 75 #91
batch_range = range(0, N_batch_files+1) if not debug_ON else range(40, 45) #(40, 45) (0, 69)

_Cmin, _Cmax = (D_randcrop-D_center)/2, (D_randcrop-D_center)/2 + D_center 
# _Cmin_middle, _Cmax_middle = (D_randcrop-D_center/2)/2, (D_randcrop-D_center/2)/2 + D_center/2 ## we only consider the middle part of the 'already_cropped' cube
            
# larger weight for the central part of the cube            
squaredDist2center_3D = np.sum((np.indices((D_center,)*3)-(D_center-1)/2.)**2, axis = 0)
gaussian_weight_3D = np.exp(-1 * squaredDist2center_3D/128.)
        
dataFolder = '/home/mengqi/dataset/MVS/lasagne/save_reconstruction_result/'
##def init_point_file(i): return '/hdd1t2/dataset/MVS/samplesVoxelVolume/init_pcds/saved_init/saved_prediction_rgb_params_model17-{}_42.npz'.format(i)
## '/hdd1t/dataset/MVS/samplesVoxelVolume/init_pcds/saved_init/saved_prediction_rgb_params_model17-18_42.npz'
#init_point_file = lambda i: dataFolder+'saved_prediction_rgb_params_model17-6viewPairs-resol0.400-strideRatio0.500-batch-{}_68.npz'.format(i)
#part_of_npz_fileName = init_point_file(0).split('_')[-2] #'model75-6viewPairs-resol0.400-strideRatio0.500-batch-0' 
init_point_file = lambda i: dataFolder+'saved_prediction_rgb_params_{}BB/model{}-{}viewPairs-resol0.400-strideRatio0.500/batch-{}_{}.npz'.format(\
        'self' if debug_ON else 'model',modelIndx, N_viewPairs,i,N_batch_files)

dataFolder_maskFile = os.path.dirname(init_point_file(0))
saved_rayPool_votes_file = os.path.join(dataFolder_maskFile, 'rayPool_votes_model{}_batch{}_{}_maxThresh{}.npz'.format(modelIndx, batch_range[0], batch_range[-1], max_adapThresh))

part_of_npz_fileName = init_point_file(0).split('/')[-2] #'model75-6viewPairs-resol0.400-strideRatio0.500' 


sigmoid_xreflect_shift = lambda x, _x_shift: 1/(1+math.exp(x - _x_shift)) 

## store the REFERENCE of the ix/iy/iz th cube's points in points_3Ddict[ix,iy,iz]. means the key (ix,iy,iz), with default value of zero array.
refine_cubes_predictionOrig_3Ddict = sparseDict(np.zeros((1,) + (D_randcrop,)*3).astype(np.float32)) 
refine_cubes_normal_3Ddict = sparseDict(np.zeros((3,) + (D_randcrop,)*3).astype(np.int8)) 
refine_cubes_surface_3Ddict = sparseDict(np.zeros((D_center,)*3).astype(np.bool)) 
refine_cubes_rgb_3Ddict = sparseDict(np.zeros((D_randcrop,)*3).astype(np.uint8)) 
# TODO: the resol / modelIndx shouldn't save so much times
refine_cubes_param_3Ddict = sparseDict(np.zeros((5,)).astype(np.float64)) # xyz/resol/modelIndx 
selected_viewPair_viewIndx_3Ddict = sparseDict(np.zeros((1,2)).astype(np.uint16)) 
N_argmax_decisions_3Ddict = sparseDict(np.zeros((1,) + (D_randcrop,)*3).astype(np.uint8)) # voxel's # of max votes along rays traced from views 
## max # cubes along each coordinates
Nx, Ny, Nz = 0, 0, 0 ## Nx * Ny * Nz = # of 3D cubes

cameraT_folder = '/home/mengqi/dataset/MVS/cameraT/'
cameraPO_folder = '/home/mengqi/dataset/MVS/pos/'
cameraTs = prepare_data.load_cameraT_as_np(cameraT_folder)
cameraPOs = prepare_data.load_cameraPos_as_np(cameraPO_folder)

# prediction_list, rgb_list, param_list, selected_viewPair_viewIndx_list = [], [], [], []

for _iter, _file_iter in enumerate(batch_range): 
    file_name = init_point_file(_file_iter)
    # if not os.path.exists(file_name):
    #     print('file not exist: {}'.format(file_name))
    #     continue
    try:
        with open(file_name) as f:
            npz_file = np.load(f)
            """
            prediction_sub: {N,1,D,D,D} float32
            rgb_sub: {N,3,D,D,D} uint8
            param_sub: {N,8} float64 # x,y,z,resol,modelIndx,indx_d0,indx_d1,indx_d2
            selected_viewPair_viewIndx_sub: {N, No_viewPairs, 2}
            """
            prediction_sub, rgb_sub, params_sub, selected_viewPair_viewIndx_sub = \
                    npz_file["prediction"], npz_file["rgb"], npz_file["param"], npz_file["selected_pairIndx"] 
            ## attention: from my experiment: the numpy operation on float16 is not as efficient as float32. at least 20X slower
            prediction_sub = prediction_sub.astype(np.float32) 
            rgb_sub = rgb_sub.astype(np.uint8)
            # finnally, only the xyz/resol/modelIndx will be stored. In case the entire params_sub will be saved in memory, we deep copy it.
            param_sub = copy.deepcopy(params_sub[:,:5]).astype(np.float64) 
            ijk_coord = copy.deepcopy(params_sub[:,5:8])
            ijk_coords = ijk_coord.astype(np.uint16)

            cubeSelector = np.amax(prediction_sub, axis=(1,2,3,4)) > 1./max_adapThresh
            print('loaded {} / {} cubes'.format(cubeSelector.sum(), cubeSelector.size))
            prediction_sub = copy.deepcopy(prediction_sub[cubeSelector])
            rgb_sub = copy.deepcopy(rgb_sub[cubeSelector])
            param_sub = copy.deepcopy(param_sub[cubeSelector])
            ijk_coords = copy.deepcopy(ijk_coords[cubeSelector])
            selected_viewPair_viewIndx_sub = copy.deepcopy(selected_viewPair_viewIndx_sub[cubeSelector])
            for _n, _ijk in enumerate(ijk_coords):
                _x,_y,_z = _ijk
                if load_origPrediction: 
                    refine_cubes_predictionOrig_3Ddict[_x, _y, _z] = prediction_sub[_n]
                else: #"loaded prediction: globMax."
                    # localmax_prediction = localmax_ndim(prediction_sub[_n], axes=[1,2,3]) # (1,D,D,D) ==> (3,1,D,D,D)
                    globmax_prediction = globmax_ndim(prediction_sub[_n], axes=[1,2,3]) # (1,D,D,D) ==> (3,1,D,D,D)
                    refine_cubes_predictionOrig_3Ddict[_x, _y, _z] = prediction_sub[_n] * (np.sum(globmax_prediction, axis=0) >= 1) 
                # refine_cubes_predictionOrig_3Ddict[_x, _y, _z] = prediction_sub[_n]
                refine_cubes_rgb_3Ddict[_x, _y, _z] = rgb_sub[_n]
                refine_cubes_param_3Ddict[_x, _y, _z] = param_sub[_n]
                selected_viewPair_viewIndx_3Ddict[_x, _y, _z] = selected_viewPair_viewIndx_sub[_n]
                Nx, Ny, Nz = max(Nx, _x + 1), max(Ny, _y + 1), max(Nz, _z + 1)
                ## calculate the Quadrant of the vector (mean_cameraO - cubeCenter)
                selected_views = selected_viewPair_viewIndx_3Ddict[_x, _y, _z].flatten()
                mean_cameraO = np.mean(cameraTs[selected_views], axis=0)
                min_x,min_y,min_z,resol,_ = refine_cubes_param_3Ddict[_x, _y, _z]
                cubeCenter = np.array([min_x,min_y,min_z]) + resol * D_randcrop / 2.
                if with_pt_normal:
                    refine_cubes_normal_3Ddict[_x, _y, _z] = localmax_prediction.squeeze() * np.sign(mean_cameraO - cubeCenter)[:,None,None,None]
     
    except:
        print('Warning: file not exist / valid: {}'.format(file_name))
            
    print _file_iter
         

# prediction, rgb, param = np.concatenate(prediction_list,axis=0), np.concatenate(rgb_list,axis=0), np.concatenate(param_list,axis=0)
# del prediction_list
# del rgb_list
# del param_list 

if enable_maskID_pool:
    
    # selected_viewPair_viewIndx_np = np.concatenate(selected_viewPair_viewIndx_list, axis=0)
    # del selected_viewPair_viewIndx_list
    N_cubes = len(refine_cubes_predictionOrig_3Ddict.keys())
    cubeD = prediction_sub.shape[-1]
    
    
    if not os.path.isfile(saved_rayPool_votes_file):  
        t = time.time()
        
        Nmax_cubes_batch = 1 # better use small value, because of non-eligent exit of python multiprocess (don't )
        # prediction_masked_list, N_argmax_decisions_list = [], []
        for _xyz_key in refine_cubes_predictionOrig_3Ddict.iterkeys():   
            N_argmax_decisions_3Ddict[_xyz_key] = rayPooling.rayPooling_1cube_numpy(cameraPOs, cameraTs, \
                    viewPair_viewIndx = selected_viewPair_viewIndx_3Ddict[_xyz_key],\
                    param = refine_cubes_param_3Ddict[_xyz_key],\
                    cube_prediction = refine_cubes_predictionOrig_3Ddict[_xyz_key],\
                    prediction_thresh = 1.0 / max_adapThresh)[None,...].astype(np.uint8)
            if check_rayPooling:
                t2 = time.time()
                _new = rayPooling.rayPooling_1cube_numpy(cameraPOs, cameraTs, \
                        viewPair_viewIndx = selected_viewPair_viewIndx_3Ddict[_xyz_key],\
                        param = refine_cubes_param_3Ddict[_xyz_key],\
                        cube_prediction = refine_cubes_predictionOrig_3Ddict[_xyz_key],\
                        prediction_thresh = 1.0 / max_adapThresh)[None,...]
                print "new: {}rayTrace pool takes: {} ".format(N_cubes, time.time()-t); t2 = time.time()
                _old = rayPooling.rayPooling_1cube_numpy(cameraPOs, cameraTs, \
                        viewPair_viewIndx = selected_viewPair_viewIndx_3Ddict[_xyz_key],\
                        param = refine_cubes_param_3Ddict[_xyz_key],\
                        cube_prediction = refine_cubes_predictionOrig_3Ddict[_xyz_key])[None,...]
                print "orig: {}rayTrace pool takes: {} ".format(N_cubes, time.time()-t)
                sys.path.append("../../3.2D_2_3D-train/2.2D_2_3D-train")
                import train_val_data
                train_val_data.visualize_N_densities_pcl([_old,_new])  

        print "{}rayTrace pool takes: {} ".format(N_cubes, time.time()-t)
            
        #prediction = prediction_masked.astype(np.float32)
            
        utils_sparseDict.save_numpy_sparseDict(saved_rayPool_votes_file, N_argmax_decisions_3Ddict)
    else:
        N_argmax_decisions_3Ddict = utils_sparseDict.load_numpy_sparseDict(saved_rayPool_votes_file)

## VISUALIZATION
if visualization_ON:
    sys.path.append("../../3.2D_2_3D-train/2.2D_2_3D-train")  # change the params_volume.py: `whatUWant = "test_fusionNet"`
    import train_val_data
    for i in range(18):
        key = refine_cubes_predictionOrig_3Ddict.keys()[i*5]
        localmax_decision = np.sum(localmax_ndim(refine_cubes_predictionOrig_3Ddict[key], axes=[1,2,3]), axis=0)
        # localmax_decision = np.sum(localmax_ndim(N_argmax_decisions_3Ddict[key], axes=[1,2,3]), axis=0)
        # localmax_1D_prediction = np.any(localmax_ndim(refine_cubes_predictionOrig_3Ddict[key]), axis=0)
        # localmax_2D_prediction = np.sum(localmax_ndim(refine_cubes_predictionOrig_3Ddict[key]), axis=0) >= 2
        # localmax_3D_prediction = np.sum(localmax_ndim(refine_cubes_predictionOrig_3Ddict[key]), axis=0) >= 3

        argmax_decision = np.sum(globmax_ndim(refine_cubes_predictionOrig_3Ddict[key], axes=[1,2,3]), axis=0)
            
        train_val_data.visualize_N_densities_pcl([\
                # refine_cubes_predictionOrig_3Ddict[key]*2, \
                (refine_cubes_predictionOrig_3Ddict[key]*2)[0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax], \
                #refine_cubes_predictionOrig_3Ddict[key]*2*(N_argmax_decisions_3Ddict[key] >= 2),\
                # refine_cubes_predictionOrig_3Ddict[key]*2*localmax_1D_prediction,\
                (refine_cubes_predictionOrig_3Ddict[key]*2*(localmax_decision>=1))[0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax],\
                (refine_cubes_predictionOrig_3Ddict[key]*2*(localmax_decision>=2))[0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax],\
                (refine_cubes_predictionOrig_3Ddict[key]*2*(localmax_decision>=3))[0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax],\
                # refine_cubes_predictionOrig_3Ddict[key]*2*(argmax_decision>0),\
                # refine_cubes_predictionOrig_3Ddict[key]*2*(argmax_decision>1),\
                # refine_cubes_predictionOrig_3Ddict[key]*2*(argmax_decision>2),\
                (refine_cubes_predictionOrig_3Ddict[key]*2*(argmax_decision>=1))[0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax],\
                (refine_cubes_predictionOrig_3Ddict[key]*2*(argmax_decision>=2))[0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax],\
                (refine_cubes_predictionOrig_3Ddict[key]*2*(argmax_decision>=3))[0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax],\
                (refine_cubes_predictionOrig_3Ddict[key]*2*(N_argmax_decisions_3Ddict[key] >= 10))[0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax] ])#,\
                #(refine_cubes_predictionOrig_3Ddict[key]*2*(N_argmax_decisions_3Ddict[key] >= 12))[0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax]]) 

# start tune params!        
# for thresh_NO_viewPairs_decisions in thresh_NO_viewPairs_decisions_list:
thresh_NO_viewPairs_decisions = thresh_NO_viewPairs_decisions_list[0]
for weight_AND_term in range(2,21,2): #[4,6,8,10]:
    save_result_fld = dataFolder + "adapt_thresh/" + part_of_npz_fileName + "/{}-{}-{}_and/".format(batch_range[0], batch_range[-1], weight_AND_term)
    if not os.path.exists(save_result_fld):
        os.makedirs(save_result_fld)
    # prediction_reload = np.copy(prediction) ## Attention, the original prediction cannot change during param tuning.
    if enable_maskID_pool:
        if refine_cubes_predictionOrig_3Ddict.viewkeys() != N_argmax_decisions_3Ddict.viewkeys():
            raise Warning('keys of N_argmax_decisions_3Ddict and refine_cubes_predictionOrig_3Ddict do not match, regenerate it.')
        refine_cubes_prediction_3Ddict = sparseDict(np.zeros((1,) + (D_randcrop,)*3).astype(np.float32)) 
        for _xyz_key in refine_cubes_predictionOrig_3Ddict.iterkeys():
            refine_cubes_prediction_3Ddict[_xyz_key] = refine_cubes_predictionOrig_3Ddict[_xyz_key] \
                    * (N_argmax_decisions_3Ddict[_xyz_key] >= thresh_NO_viewPairs_decisions)
        print 'thresh_NO_viewPairs_decisions: {}'.format(thresh_NO_viewPairs_decisions)
    else:
        refine_cubes_prediction_3Ddict = refine_cubes_predictionOrig_3Ddict

    for init_adapThresh in init_adapThresh_list:
            
        print 'init_adapThresh: {}'.format(init_adapThresh)
    
        # TODO: the refine_cubes_occupancy_np is totally useless, latter can use refine_cubes_surface_3Ddict.iterkeys() to read out the occupied cube coordinates
        refine_cubes_occupancy_np = np.zeros((Nx,Ny,Nz)).astype(np.bool)
        ## adaptive threshld; # of nonempty neighbors; # of connected cubes

        
        refine_adapThresh_np = np.zeros((Nx,Ny,Nz))
        refine_adapThresh_np[:] = init_adapThresh

        _return_surface_from_predictList_and_oldThreshList = lambda _x,_y,_z: (\
                refine_cubes_prediction_3Ddict[_x, _y, _z][0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax]*refine_adapThresh_np[_x,_y,_z])>=1
        _return_surface_from_predictList_and_newThresh = lambda _x,_y,_z,_thresh: (\
                refine_cubes_prediction_3Ddict[_x, _y, _z][0,_Cmin:_Cmax,_Cmin:_Cmax,_Cmin:_Cmax]*_thresh)>=1

        print 'loaded the initial point cloud data'    
        for _xyz_withPrediction in refine_cubes_prediction_3Ddict.iterkeys():

            _x, _y, _z = _xyz_withPrediction
            refine_cubes_surface_3Ddict[_x, _y, _z] = _return_surface_from_predictList_and_oldThreshList(_x,_y,_z)
            refine_cubes_occupancy_np[_x,_y,_z] = refine_cubes_surface_3Ddict[_x, _y, _z].sum() != 0
        
        #################
        print 'start updating the threshold'
        refine_adapThresh_np_new = np.copy(refine_adapThresh_np)
        # store the normal information of halfCube into dict, to avoid multiple calculation.
        # because the surface-probability is fixed. If the threshold is fixed, the 0/1 surface will fix
        # then the normal information of the half cubes will be fixed.
        # So store the normal information by the keys (x,y,z,thresh,xyz_relativeHalf)
        # where: xyz_relativeHalf = (+-1,0,0)/(0,+-1,0)/(0,0,+-1)
        normalInformation_3Ddict = sparseDict(0) 

        def _cube_surface_percent(_x,_y,_z) : return refine_cubes_surface_3Ddict[_x, _y, _z].sum() / (1. * D_center**3)
        # def _cube_middle_percent(_x,_y,_z) : return ((refine_cubes_prediction_3Ddict[_x, _y, _z][0,_Cmin_middle:_Cmax_middle,_Cmin_middle:_Cmax_middle,\
        #                                     _Cmin_middle:_Cmax_middle] * refine_adapThresh_np[_x,_y,_z])>=1).sum() / (1. * (D_center/2)**3)

        for _iter in xrange(N_refine_iter):
            if RGB_tmp_results:
                # refresh to the orig color at the begining of each iter
                refine_cubes_rgb_3Ddict_4visual = copy.deepcopy(refine_cubes_rgb_3Ddict)  
            # _iter_xyz = np.argwhere(refine_cubes_occupancy_np==True) # skip the empty cubes
            
            for _xyz_withSurface in refine_cubes_surface_3Ddict.iterkeys():
                _x,_y,_z = _xyz_withSurface

                # ***********elementwise update***********
                ## initialize low scale value, elementwise update tries to eliminate floating noise
                ##if _cube_surface_percent < .3 :
                    ##refine_adapThresh_np_new[_x,_y,_z] -= .1
                
                # *********pairwise update***********
                neighbor_indx_shift=np.asarray([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]]) ## 6 neighbors
                votes_from_neighb = 0
                counter_4_neigh_middle_occup = 0

                element_cost=np.array([0,0,0])

                for _xyz_shift in neighbor_indx_shift:
                    _x_ovlp,_y_ovlp,_z_ovlp = np.asarray([_x,_y,_z])+_xyz_shift
                    ## consider the strideRatio = 1 neighboring cubes, no overlapping with the current cube
                    _x_adjac,_y_adjac,_z_adjac = np.asarray([_x,_y,_z])+_xyz_shift * 2
                    
                    ## the neighbor doesn't exist
                    if _x_adjac>=Nx or _y_adjac>=Ny or _z_adjac>=Nz: 
                        continue
                    # # otherwise the argument of method 'access_half_cube_3D' could have [] 
                    # if not refine_cubes_occupancy_np[_x_ovlp,_y_ovlp,_z_ovlp]: 
                    #     refine_cubes_surface_3Ddict[_x_ovlp, _y_ovlp, _z_ovlp] = np.zeros((D_center,)*3)
                    # halfCube_gauss = access_half_cube_3D(gaussian_weight_3D, _xyz_shift * -1)

                    # half cube of the neighboring cube WITH overlapping
                    halfCube_ovlp = access_half_cube_3D(refine_cubes_surface_3Ddict[_x_ovlp, _y_ovlp, _z_ovlp], _xyz_shift * -1) 
                    _thresh_ovlp = refine_adapThresh_np[_x_ovlp, _y_ovlp, _z_ovlp]
                    # half cube of the neighboring cube WITHOUT overlapping
                    halfCube_adjac = access_half_cube_3D(refine_cubes_surface_3Ddict[_x_adjac,_y_adjac,_z_adjac], _xyz_shift * -1) 
                    _thresh_adjac = refine_adapThresh_np[_x_adjac,_y_adjac,_z_adjac]

                    threshold_perturbation_list = [-0.1, 0, 0.1]

                    if old_adpthresh_policy: 
                        '''######## pseudo code of solving
                        ##      thresh_new = thresh
                        ##      for each iteration 
                        ##          for ith cube
                        ##              cost_i = [0,0,0] # corresponding to decrease/keep/increase the threshold
                        ##              for _n, _threshold_perturbation in [-0.1,0,0.1]
                        ##                  for jth overlapping neighboring cube & kth adjcent neighboring cube: 
                        ##                      surf_j = thresholding(surface_probability_j, thresh_j)
                        ##                      surf_i = thresholding(surface_probability_i, thresh_i + _threshold_perturbation)
                        ##                      cost_i[_n] += overlapping_XOR(surf_i, surf_j)
                        ##                      if surf_i, surf_j have surface in the overlappint half part:
                        ##                          normal_i/j = normal vector for the overlappint half part
                        ##                          normal_consistency_ovlp = cos(<normal_i, normal_j>)
                        ##                          cost_i[_n] -= overlapping_AND(surf_i, surf_j) * normal_consistency_ovlp * desired_thickness
                        ##                      if surf_i, surf_k have surface in the adjacent half cube:
                        ##                          normal_k = normal vector for the overlappint half part
                        ##                          normal_consistency_adjc = cos(<normal_i, normal_k>)
                        ##                          cost_i[_n] -= adjacent_OR(surf_i, surf_k) * normal_consistency_adjc
                        ##              thresh_new_i = thresh_i + [-0.1,0,0.1][argmin(cost_i)]
                        ##          thresh = thresh_new 
                        ##
                        ##
                        ## where surf_i/j = thresholding(surface_probability_i/j, thresh_i/j)
                        ##       cube_j is 6 / 8 cube neighbor of cube_i
                                
                        ## given: min thresh = 1.2
                        ##        init/max thresh = 2 # because the finnal cost function has the range like: [0,.1,.5,.3,0,-.2,-.5,-.4,-.3,...,-.01], in order to minimize the cost, should initiate from high threshold. 
                        ##
                            Finnal cost = overlapping_XOR - w1 * overlapping_AND - w2 * adjacent_OR
                            w1 = normal_consistency_ovlp * desired_thickness
                            w2 = normal_consistency_adjc
                            Finnal cost range: [0,.1,.5,.3,0,-.2,-.5,-.4,-.3,...,-.01] with the increase of threshold from 0-->2
                            For real surface, the optimal threshold is around 1.5
                                floating noise, ... 0
                                connecting noise, ... 0
                        ## 
                            overlapping_XOR cost: penanty dissimilarity of overlapping part (all the cubes --> empty)
                                                this term's range like: [0, .1, .5, .3, .2, .1, ... , .01]
                            -1 * overlapping_AND cost: encourage surface overlapping of overlapping cube surface
                                                range: [0, 0, -.1, -.3, -.5, -.8, -.9]
                            -1 * adjacent_OR cost: encourage connectivity of adjacent surface
                                                range: [0, -.1, -.3, -.5, -.8, -.9, -.95]
                            desired_thickness: sigmoid with range (2,0): stimulate thin surface to increase threshold, penalize thick ones.
                                                Without this, some thick surface come out. [1.9, 1.6, 1, .4, .1]
                            normal_consistency_adjc/ovlp: encourage normal consistency, will suppress the connecting noise.
                                                
                        '''
                        pass 
                        # for _n_thresh, _threshold_shift in enumerate(threshold_perturbation_list):
                        #     _thresh_current = refine_adapThresh_np[_x,_y,_z] + _threshold_shift
                        #     halfCube_current = access_half_cube_3D(_return_surface_from_predictList_and_newThresh(_x,_y,_z,\
                        #             _thresh_current), _xyz_shift)
                        #     # element_cost[_n_thresh] += ((halfCube_current != halfCube_ovlp) * halfCube_gauss).sum() # overlappint_xor 
                        #     element_cost[_n_thresh] += (halfCube_current != halfCube_ovlp).sum() # overlappint_xor 

                        #     if halfCube_current.sum()>=6: 
                        #         normal_vect_current, thickness_current = get_normalInfo_from_dict(normalInformation_3Ddict, (_x,_y,_z,_thresh_current)+tuple(_xyz_shift), halfCube_current)

                        #         #-------------------- consider the strideRatio = 0.5 neighboring cube (_ovlp)
                        #         if halfCube_ovlp.sum()>=6:
                        #             normal_vect_ovlp, thickness_ovlp = get_normalInfo_from_dict(normalInformation_3Ddict, (_x_ovlp, _y_ovlp, _z_ovlp,_thresh_ovlp)+tuple(_xyz_shift * -1), halfCube_ovlp)
                        #             cos_normal_current_ovlp = abs(np.dot(normal_vect_current, normal_vect_ovlp.T)) # abs(dot product) of 2 normal vectors
                                    
                        #             # weight_ovlp_OR_term = cos_normal_current_ovlp * \
                        #             #         2.0 * sigmoid_xreflect_shift(thickness_current , _x_shift = desired_thickness) * \
                        #             #         2.0 * sigmoid_xreflect_shift(thickness_ovlp , _x_shift = desired_thickness) # make the sigmoid weight output 1 around desired thickness, so 2.0 *
                        #             # # element_cost[_n_thresh] -= (((halfCube_current == 1) | (halfCube_ovlp == 1)) * halfCube_gauss).sum() * weight_ovlp_OR_term # overlapping_or
                        #             # # element_cost[_n_thresh] -= ((halfCube_current == 1) | (halfCube_ovlp == 1)).sum() * weight_ovlp_OR_term # overlapping_or

                        #             weight_AND_term = cos_normal_current_ovlp * \
                        #                     2.0 * sigmoid_xreflect_shift(thickness_current , _x_shift = desired_thickness) * \
                        #                     2.0 * sigmoid_xreflect_shift(thickness_ovlp , _x_shift = desired_thickness) # make the sigmoid weight output 1 around desired thickness, so 2.0 *
                        #             element_cost[_n_thresh] -= ((halfCube_current == 1) & (halfCube_ovlp == 1)).sum() * weight_AND_term # overlapping_and

                        #         #-------------------- consider the strideRatio = 1 neighboring cube (_adjac)
                        #         if halfCube_adjac.sum()>=6:
                        #             normal_vect_adjac, thickness_adjac = get_normalInfo_from_dict(normalInformation_3Ddict, (_x_adjac,_y_adjac,_z_adjac,_thresh_adjac)+tuple(_xyz_shift * -1), halfCube_adjac)
                        #             cos_normal_current_adjac = abs(np.dot(normal_vect_current, normal_vect_adjac.T)) # dot product of 2 normal vectors

                        #             # 0.5 is the importance of the cube according to the distance.
                        #             weight_adjac_OR_term = cos_normal_current_adjac * 0.5 
                        #             # element_cost[_n_thresh] -= (((halfCube_current == 1) | (halfCube_adjac == 1)) * halfCube_gauss).sum() * weight_ovlp_OR_term # overlapping_or
                        #             element_cost[_n_thresh] -= ((halfCube_current == 1) | (halfCube_adjac == 1)).sum() * weight_adjac_OR_term # overlapping_or
                        #             # element_cost[_n_thresh] -= (halfCube_current == 1).sum() * weight_adjac_OR_term # overlapping_or

                        #         #     # weight_densityDiff_term = cos_normal_current_adjac * \
                        #         #     #         2.0 * sigmoid_xreflect_shift(thickness_current , _x_shift = desired_thickness) * \
                        #         #     #         2.0 * sigmoid_xreflect_shift(thickness_adjac , _x_shift = desired_thickness) # make the sigmoid weight output 1 around desired thickness, so 2.0 *
                        #         #     # # densityDiff_term penalize case: density_current - density_adjac > 0
                        #         #     # # densityDiff_term encourage case: density_current - density_adjac < 0
                        #         #     # element_cost[_n_thresh] -= (halfCube_adjac.sum() - halfCube_current.sum()) * weight_densityDiff_term 
                    
                    else:   # not old_adpthresh_policy:
                        for _n_thresh, _threshold_shift in enumerate(threshold_perturbation_list):
                            _thresh_current = refine_adapThresh_np[_x,_y,_z] + _threshold_shift
                            halfCube_current = access_half_cube_3D(_return_surface_from_predictList_and_newThresh(_x,_y,_z,\
                                    _thresh_current), _xyz_shift)
                            element_cost[_n_thresh] += (halfCube_current != halfCube_ovlp).sum() # overlappint_xor 

                            if halfCube_current.sum()>=6: 
                                # normal_vect_current, var_current = get_normalInfo_from_dict(normalInformation_3Ddict, (_x,_y,_z,_thresh_current)+tuple(_xyz_shift), halfCube_current)

                                # # if the 2nd / 3rd variance < 5, most likely this half cube is only noise rather than a surface
                                # # if the normal vector is point to the adjacent cube, continue. Because no matter how the adjacent cube's surface cannot connect with current cube's surface.
                                # if ((var_current[1] / var_current[-1]) < 5) \
                                #         and (np.argmax(np.abs(_xyz_shift)) == np.argmax(np.abs(normal_vect_current))):
                                #     continue

                                #-------------------- consider the strideRatio = 0.5 neighboring cube (_ovlp)
                                if halfCube_ovlp.sum()>=6:
                                    # normal_vect_ovlp, var_ovlp = get_normalInfo_from_dict(normalInformation_3Ddict, (_x_ovlp, _y_ovlp, _z_ovlp,_thresh_ovlp)+tuple(_xyz_shift * -1), halfCube_ovlp)
                                    # # cos_normal_current_ovlp = abs(np.dot(normal_vect_current, normal_vect_ovlp.T)) # abs(dot product) of 2 normal vectors
                                    # if ((var_ovlp[1] / var_ovlp[-1]) < 5) \
                                    #         and (np.argmax(np.abs(_xyz_shift)) == np.argmax(np.abs(normal_vect_ovlp))):
                                    #     continue
                                    
                                    element_cost[_n_thresh] -= ((halfCube_current == 1) & (halfCube_ovlp == 1)).sum() * weight_AND_term # overlapping_and

                            # #     #-------------------- consider the strideRatio = 1 neighboring cube (_adjac)
                            #     if halfCube_adjac.sum()>=6:
                            #         normal_vect_adjac, var_adjac = get_normalInfo_from_dict(normalInformation_3Ddict, (_x_adjac,_y_adjac,_z_adjac,_thresh_adjac)+tuple(_xyz_shift * -1), halfCube_adjac)
                            #         if ((var_adjac[1] / var_adjac[-1]) < 5) \
                            #                 and (np.argmax(np.abs(_xyz_shift)) == np.argmax(np.abs(normal_vect_adjac))):
                            #             continue
                            #         cos_normal_current_adjac = abs(np.dot(normal_vect_current, normal_vect_adjac.T)) # dot product of 2 normal vectors

                            # #         # 0.5 is the importance of the cube according to the distance.
                            #         weight_adjac_OR_term = cos_normal_current_adjac * 0.5 
                            #         element_cost[_n_thresh] -= ((halfCube_current == 1) | (halfCube_adjac == 1)).sum() * weight_adjac_OR_term # overlapping_or


                if RGB_tmp_results: 
                    # visualization
                    # R/G/B to visualize 3 cases: thresh += [-0.1/0/0.1]
                    # red represents to reduce thresh!
                    refine_cubes_rgb_3Ddict_4visual[_x, _y, _z][np.argmin(element_cost),...] = 255  # (3,D,D,D) RGB
                    
                refine_adapThresh_np_new[_x,_y,_z] = refine_adapThresh_np[_x,_y,_z] + threshold_perturbation_list[np.argmin(element_cost)] 
                refine_adapThresh_np_new[_x,_y,_z] = max(refine_adapThresh_np_new[_x,_y,_z], min_adapThresh) 


                            
            print 'updated iteration {}'.format(_iter)
            refine_adapThresh_np = np.copy(refine_adapThresh_np_new)  

            for _xyz_withPrediction in refine_cubes_prediction_3Ddict.iterkeys():
                _x,_y,_z = _xyz_withPrediction
                refine_cubes_surface_3Ddict[_x, _y, _z] = _return_surface_from_predictList_and_oldThreshList(_x,_y,_z)                     
                refine_cubes_occupancy_np[_x,_y,_z] = refine_cubes_surface_3Ddict[_x, _y, _z].sum() != 0


            _kwargs = {'refine_cubes_prediction_3Ddict':refine_cubes_prediction_3Ddict, \
                    'refine_adapThresh_np': refine_adapThresh_np, \
                    'refine_cubes_occupancy_np': refine_cubes_occupancy_np, \
                    'refine_cubes_rgb_3Ddict': refine_cubes_rgb_3Ddict, \
                    'refine_cubes_param_3Ddict': refine_cubes_param_3Ddict, \
                    'filename': 'adapThresh_rayPool_{}-init{}_Ndecision{}_iter{}_realColor'.format('On' if enable_maskID_pool else 'Off',init_adapThresh,thresh_NO_viewPairs_decisions,_iter), \
                    'pcd_folder': save_result_fld, \
                    'center_D_range': [_Cmin, _Cmax-1], \
                    'refine_cubes_normal_3Ddict': refine_cubes_normal_3Ddict, \
                    'with_pt_normal': with_pt_normal}
            
            try:
                if not debug_ON:
                    raise Warning("save ply in new process could lead to memoryError.")
                Process(target=refine_save_points_2_ply, kwargs=_kwargs).start()
                print('refine_save_points_2_ply in new process')
            except:
                refine_save_points_2_ply(**_kwargs)
            # refine_save_points_2_ply(refine_cubes_prediction_3Ddict, refine_adapThresh_np, refine_cubes_occupancy_np, refine_cubes_rgb_3Ddict, refine_cubes_param_3Ddict, \
            #      'adapThresh_rayPool_{}-init{}_Ndecision{}_iter{}_realColor'.format('On' if enable_maskID_pool else 'Off',init_adapThresh,thresh_NO_viewPairs_decisions,_iter),\
            #      pcd_folder=save_result_fld, center_D_range = [_Cmin, _Cmax-1], refine_cubes_normal_3Ddict = refine_cubes_normal_3Ddict, with_pt_normal = with_pt_normal)
            if RGB_tmp_results:
                _kwargs['refine_cubes_rgb_3Ddict'] = refine_cubes_rgb_3Ddict_4visual
                _kwargs['filename'] = 'adapThresh_rayPool_{}-init{}_Ndecision{:.2}_iter{}_visualize_update'.format('On' if enable_maskID_pool else 'Off',init_adapThresh,thresh_NO_viewPairs_decisions,_iter)
                try:
                    if not debug_ON:
                        raise Warning("save ply in new process could lead to memoryError.")
                    Process(target=refine_save_points_2_ply, kwargs=_kwargs).start()
                    print('refine_save_points_2_ply in new process')
                except:
                    refine_save_points_2_ply(**_kwargs)
                # refine_save_points_2_ply(refine_cubes_prediction_3Ddict, refine_adapThresh_np, refine_cubes_occupancy_np, refine_cubes_rgb_3Ddict_4visual, refine_cubes_param_3Ddict, \
                #     'adapThresh_rayPool_{}-init{}_Ndecision{}_iter{}_visualize_update'.format('On' if enable_maskID_pool else 'Off',init_adapThresh,thresh_NO_viewPairs_decisions,_iter),\
                #     pcd_folder=save_result_fld, center_D_range = [_Cmin, _Cmax-1], refine_cubes_normal_3Ddict = refine_cubes_normal_3Ddict, with_pt_normal = with_pt_normal)
            print '.'
        
        print 'done'
        
