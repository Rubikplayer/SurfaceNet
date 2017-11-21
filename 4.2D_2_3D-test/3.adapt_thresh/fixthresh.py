import matlab.engine as mat # weird: cannot be imported after numpy !
import numpy as np
import copy
import sparseCubes
import os
import time


def access_partial_Occupancy_ijk(Occ_ijk, shift, D_cube):
    """
    access 1/2, 1/4, 1/8 of the cube's Occupancy_ijk, 
    at the same time, translate the origin to the shifted array. 
    For example: ijk [1,2,4] of cube with shape1 = (6,6,6)
            Its partial cube with shift = (0,-1,1) has shape2 = (6,3,3)
            Its shifted ijk --> [1, 2, 4 - 3] --> [1,2,1] (within boarder of shape2)

    Note:
        This method will use the reference of Occ_ijk, and CHANG it.

    -------------
    inputs:
        Occ_ijk: np.uint (N_voxel, 2/3/...)
        shift: np/tuple/list (2/3/..., ), specify which part of the array will be selected
                (1,-1,0): np.s_[D_mid:, :D_mid, :]
        D_cube: Occ_ijk's upper bound
    -------------
    outputs:
        Occ_partial: np.uint (n_voxel, 2/3/...)
    -------------
    example:
    >>> Occ_ijk = np.array([[1,5,2], [5,2,0], [0,1,5], [2,1,1], [4,5,5]])
    >>> gt = Occ_ijk[2:3]-np.array([[0,0,3]]) 
    >>> resul = access_partial_Occupancy_ijk(Occ_ijk, (-1,0,1), D_cube=6)
    >>> np.allclose(gt, resul) and (gt.shape == resul.shape) # because allclose(array, empty)=True !
    True
    >>> np.array_equal(Occ_ijk[1:4,:2], \
         access_partial_Occupancy_ijk(Occ_ijk[:,:2], (0,-1), D_cube=6))
    True
    """
    D_mid = D_cube / 2
    N_voxel, n_dim = Occ_ijk.shape
    select_ijk = np.ones(shape=(N_voxel,n_dim))
    for _dim in range(n_dim):
        if shift[_dim] == -1:
            select_ijk[:,_dim] = (Occ_ijk[:,_dim] >= 0) & (Occ_ijk[:,_dim] < D_mid)
        elif shift[_dim] == 0:
            select_ijk[:,_dim] = (Occ_ijk[:,_dim] >= 0) & (Occ_ijk[:,_dim] < D_cube)
        elif shift[_dim] == 1:
            select_ijk[:,_dim] = (Occ_ijk[:,_dim] >= D_mid) & (Occ_ijk[:,_dim] < D_cube)
            Occ_ijk[:, _dim] -= D_mid
        else:
            raise Warning("shift only support 3 values: -1/0/1, but got {}".format(shift))
    select_ijk = select_ijk.all(axis = 1) # (N_voxel, n_dim) --> (N_voxel,)
    return Occ_ijk[select_ijk]

def sparseOccupancy_AND_XOR(Occ1, Occ2):
    """
    perform AND or XOR operation between 2 occupancy index arrays (only with ijk of occupied indexes)

    -------------
    inputs:
        Occ1: np.uint (n1,2/3/...)
        Occ2: np.uint (n2,2/3/...)
    -------------
    outputs:
        resul_AND: how many overlapping elements 
        resul_XOR: how many non_overlapping elements
    -------------
    example:
    >>> ijk1=np.array([[1,0],[2,3],[222,666],[0,0]])
    >>> ijk2=np.array([[11,10],[2,3],[22,66],[0,0],[7,17]]) 
    >>> sparseOccupancy_AND_XOR(ijk1,ijk2)
    (2, 5)
    """
    n1, ndim1 = Occ1.shape
    n2, ndim2 = Occ2.shape
    if (n1 == 0) or (n2 == 0):
        resul_AND = 0
    else:
        Occ1_1D = Occ1.view(dtype=Occ1.dtype.descr * ndim1)
        Occ2_1D = Occ2.view(dtype=Occ2.dtype.descr * ndim2)
        resul_AND = np.intersect1d(Occ1_1D, Occ2_1D).size
    resul_XOR = n1 + n2 - resul_AND * 2
    return resul_AND, resul_XOR




def adapthresh(modelIndx, N_viewPairs, N_refine_iter, \
        init_probThresh, min_probThresh, max_probThresh,\
        rayPool_thresh, weight_AND_term_list, D_cube, \
        dataFolder, npz_file,\
        RGB_visual_ply=True, ply_before_adapthresh=True):
        
    data = sparseCubes.load_sparseCubes(npz_file)
    prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
            cube_ijk_np, param_np, viewPair_np = data

    ## before adapthresh, with init_probThresh and rayPool_thresh
    vxl_leftIndx_init_list = sparseCubes.filter_voxels(vxl_leftIndx_list=[],prediction_list=prediction_list, prob_thresh=init_probThresh,\
            rayPooling_votes_list=rayPooling_votes_list, rayPool_thresh=rayPool_thresh)
    if ply_before_adapthresh:
        save_result_fld = os.path.join(dataFolder, "results_adapthresh/model{}-{}viewPairs/".format(modelIndx, N_viewPairs))
        if not os.path.exists(save_result_fld):
            os.makedirs(save_result_fld)
        sparseCubes.save_sparseCubes_2ply(vxl_leftIndx_init_list, vxl_ijk_list, rgb_list, param_np, \
                ply_filePath=os.path.join(save_result_fld, '{:.2}_{}.ply'.format(init_probThresh, rayPool_thresh)), normal_list=None)

    neigh_shifts = np.asarray([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]]).astype(np.int8)
    thresh_perturb_list = [0.1, 0, -0.1] # note: the order matters, make sure argmin(cost) will first hit 0 before the perturb which can enlarge the pcd.
    # # partial cube access for **dense cube**. For sparse cube: 'sparseOccupancy_AND_XOR'
    # slices_3types = np.s_[:D_cube/2, :, D_cube/2:] # corresponding to slices in 3 cases: [-1, 0, 1]
    # partial_cube_slc = lambda shift: tuple([slices_3types[_d] for _d in shift + 1]) # [-1,1,0] --> (slcs[0],slcs[2],slcs[1])

    # cube_ijk2indx = {tuple(_ijk): _n for _n, _ijk in enumerate(cube_ijk_np)}
    cube_ijk2indx = {}
    for _n, _ijk in enumerate(cube_ijk_np):
        if vxl_leftIndx_init_list[_n].sum() > 0:
            cube_ijk2indx.update({tuple(_ijk): _n})

    for weight_AND_term in weight_AND_term_list:
        save_result_fld = os.path.join(save_result_fld, "{}_ANDweight".format(weight_AND_term))
        if not os.path.exists(save_result_fld):
            os.makedirs(save_result_fld)

        vxl_leftIndx_list = copy.deepcopy(vxl_leftIndx_init_list)
        occupied_vxl = lambda indx, thresh_shift: sparseCubes.filter_voxels([copy.deepcopy(vxl_leftIndx_list[indx])],\
                [prediction_list[indx]], probThresh_list[indx] + thresh_shift)[0]
        probThresh_list = [init_probThresh] * len(prediction_list)
        update_probThresh_list = copy.deepcopy(probThresh_list)
        for _iter in range(N_refine_iter): # each iteration of the algorithm
            time_iter = time.time()
            if RGB_visual_ply:
                tmp_rgb_list = copy.deepcopy(rgb_list)
            for _ijk in cube_ijk_np:
                if not cube_ijk2indx.has_key(tuple(_ijk)):
                    continue # this cube is filtered in the very beginning
                i_current = cube_ijk2indx[tuple(_ijk)]
                element_cost = np.array([0,0,0]).astype(np.float16)
                for _ijk_shift in neigh_shifts:
                    ijk_ovlp = _ijk + _ijk_shift
                    # ijk_adjc = _ijk + 2 * _ijk_shift
                    exist_ovlp = cube_ijk2indx.has_key(tuple(ijk_ovlp))
                    # exist_adjc = cube_ijk2indx.has_key(tuple(ijk_adjc))
                    if exist_ovlp:
                        i_ovlp = cube_ijk2indx[tuple(ijk_ovlp)]
                        tmp_occupancy_ovlp = vxl_ijk_list[i_ovlp][occupied_vxl(i_ovlp, 0)] # this will be changed in the next func.
                        partial_occ_ovlp = access_partial_Occupancy_ijk(Occ_ijk=tmp_occupancy_ovlp, \
                                shift=_ijk_shift*-1, D_cube = D_cube)
                    else:
                        partial_occ_ovlp = np.empty((0,3), dtype=np.uint8)
                    for _n_thresh, _thresh_perturb in enumerate(thresh_perturb_list):
                        tmp_occupancy_current = vxl_ijk_list[i_current][occupied_vxl(i_current, _thresh_perturb)]# this will be changed in the next func.
                        partial_occ_current = access_partial_Occupancy_ijk(Occ_ijk=tmp_occupancy_current, \
                                shift=_ijk_shift, D_cube = D_cube)
                        ovlp_AND, ovlp_XOR = sparseOccupancy_AND_XOR(partial_occ_current, partial_occ_ovlp)
                        element_cost[_n_thresh] += ovlp_XOR
                        if partial_occ_current.shape[0] >= 6:
                            if partial_occ_ovlp.shape[0] >= 6:
                                element_cost[_n_thresh] -= weight_AND_term * ovlp_AND

                update_probThresh_list[i_current] = probThresh_list[i_current] + thresh_perturb_list[np.argmin(element_cost)]
                update_probThresh_list[i_current] = min(update_probThresh_list[i_current], max_probThresh)

                if RGB_visual_ply:
                    tmp_rgb_list[i_current][:,np.argmin(element_cost)] = 255 # R/G/B --> threshold perturbation [-.1, 0, .1]
            probThresh_list = copy.deepcopy(update_probThresh_list)

            vxl_leftIndx_list = sparseCubes.filter_voxels(vxl_leftIndx_list=vxl_leftIndx_list,prediction_list=prediction_list, prob_thresh=probThresh_list,\
                    rayPooling_votes_list=None, rayPool_thresh=None)
            ply_filePath = os.path.join(save_result_fld, 'iter{}_adapthresh.ply'.format(_iter))
            sparseCubes.save_sparseCubes_2ply(vxl_leftIndx_list, vxl_ijk_list, rgb_list, \
                    param_np, ply_filePath=ply_filePath, normal_list=None)
            if RGB_visual_ply:
                sparseCubes.save_sparseCubes_2ply(vxl_leftIndx_list, vxl_ijk_list, tmp_rgb_list, \
                        param_np, ply_filePath=os.path.join(save_result_fld, 'iter{}_tmprgb.ply'.format(_iter)), normal_list=None)
            print 'updated iteration {}. It took {}s'.format(_iter, time.time() - time_iter)
    return ply_filePath


def fixthreshold(modelIndx, N_viewPairs,  \
        init_probThresh, \
        rayPool_thresh,  \
        dataFolder, npz_file):
        
    data = sparseCubes.load_sparseCubes(npz_file)
    prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
            cube_ijk_np, param_np, viewPair_np = data

    ## before adapthresh, with init_probThresh and rayPool_thresh
    vxl_leftIndx_init_list = sparseCubes.filter_voxels(vxl_leftIndx_list=[],prediction_list=prediction_list, prob_thresh=init_probThresh,\
            rayPooling_votes_list=rayPooling_votes_list, rayPool_thresh=rayPool_thresh)
    save_result_fld = os.path.join(dataFolder, "results_fixthresh/model{}-{}viewPairs/".format(modelIndx, N_viewPairs))
    if not os.path.exists(save_result_fld):
        os.makedirs(save_result_fld)
    ply_filePath = os.path.join(save_result_fld, 't{:.2}_rp{}.ply'.format(init_probThresh, rayPool_thresh))
    sparseCubes.save_sparseCubes_2ply(vxl_leftIndx_init_list, vxl_ijk_list, rgb_list, param_np, \
            ply_filePath=ply_filePath, normal_list=None)
    return ply_filePath




import doctest
doctest.testmod()
if __name__ == "__main__":

    import time
    import itertools

    eng = mat.start_matlab()
    matlab_command_list = []
    eng.cd('/home/mengqi/dataset/MVS/SampleSet/Matlab evaluation code')
    
    modelIndx_list = [9]
    N_viewPairs_list = [5] #range(9,0,-2)#[1,3,5,7,9]:
    # for simplicity, in order to run the matlab evaluation, reload the .npz file for different AND_weight
    rayPool_thresh_list = [0,5,10]
    init_probThresh_list = [.7, .5, .9]

    for params in itertools.product(modelIndx_list, N_viewPairs_list, rayPool_thresh_list, init_probThresh_list):
        print 'modelIndx, N_viewPairs, weight_AND_term: ', params
        modelIndx, N_viewPairs, rayPool_thresh, init_probThresh = params
        dataFolder = '/home/mengqi/dataset/MVS/lasagne/iccv_cameraReady-reconstruction_result/weighted_ON-randView_OFF-49views-cubeSize32/results_cubes_modelBB/'
        npz_file = os.path.join(dataFolder,'model{0}-{1}viewPairs.npz'.format(modelIndx, N_viewPairs))

        kwargs = {'init_probThresh': init_probThresh, \
                'modelIndx': modelIndx, 'N_viewPairs': N_viewPairs, \
                'rayPool_thresh': rayPool_thresh, \
                'dataFolder': dataFolder, 'npz_file': npz_file }
        try: 
            ply_filePath = fixthreshold(**kwargs)
            async_command = eng.eval_ply(modelIndx, ply_filePath, ply_filePath.replace('.ply','.mat'), async = True)
            matlab_command_list.append([ply_filePath, async_command])
            print('async eval_ply ing.')
        except Exception, err:
            print('exception of the func adapthresh', Exception, err)
    

    try:
        while len(matlab_command_list):
            for _i, [_ply_file, _command] in enumerate(matlab_command_list):
                if _command.done():
                    print _ply_file, _command.result()
                    del(matlab_command_list[_i])
            time.sleep(.1)
        print('async eval_ply done.')
    except Exception, err:
        print('async eval_ply raised error.', Exception, err)

    eng.exit()        
    print('the matlab engine is exited.')



