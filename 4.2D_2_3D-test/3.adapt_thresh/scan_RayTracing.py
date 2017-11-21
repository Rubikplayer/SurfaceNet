
from train_val_data_adaptThresh import *
import numpy as np
import time

debug_ON = False
const = 10
print_output_ON = False

define_theano_pool_ON = debug_ON if not debug_ON else False 
define_numpy_pool_ON = debug_ON if not debug_ON else True
define_npSort_pool_ON = debug_ON if not debug_ON else False
define_multiprocess_pool_ON = debug_ON if not debug_ON else True
define_process_each_maskID_gnumpy_ON = debug_ON if not debug_ON else False

if define_theano_pool_ON:
    import theano
    import theano.tensor as T
    #tensor_mask = T.arange(25).reshape((5,5)).astype(T.)
    #tensor_value = T.arange(25).reshape((5,5))
    tensor_mask = T.TensorType('int32',(False,)*5)('mask')
    tensor_value = T.TensorType('float32',(False,)*5)('value')



#----------------------------------------------------------------------
def generate_maskIDCubes_given_viewIndx(cameraPOs_np, cubeD, pts_4D, view_viewIndx):
    """
    generate maskIDCubes given viewIndx
    inputs:
    cubeD: cube size [cubeD,cubeD,cubeD]
    pts_4D: [4,cubeD**3]
    view_viewIndx: e.g. [1,1,3,5,4,3]
    return:
    maskIDCube: [len(view_viewIndx),1,cubeD,cubeD,cubeD]
    """
    ## because view_viewIndx could have redundant indxes, [1,1,3,5,4,3] --> only calculate [1,3,4,5]'s maskIDCube
    # TODO, finnaly only the certer part of the cube will be left, we only need to save/refine those voxels.
    pixIDmask_cubes = np.zeros((max(view_viewIndx)+1,1,cubeD,cubeD,cubeD), dtype=np.uint16)
    for _viewIndx in set(view_viewIndx):
        # perspective projection
        projection_M = cameraPOs_np[_viewIndx]
        pts_3D = np.dot(projection_M, pts_4D) 
        pts_3D[:-1] /= pts_3D[-1] # the result is vector: [w,h,1], w is the first dim!!!
        pts_2D = pts_3D[:-1].round().astype(np.uint16)
        pts_w, pts_h = pts_2D[0], pts_2D[1]
        pts_pixID = map_pixHW_2_uintIDnp(pts_w, pts_h)
        pixIDmask_cubes[_viewIndx] = pts_pixID.reshape((1,cubeD,cubeD,cubeD))        
    return pixIDmask_cubes[view_viewIndx] # should use pixIDmask_cubes[set(view_viewIndx)], which however leads to unfixed np shape

#----------------------------------------------------------------------
def generate_maskIDCubes(cameraPOs, viewPair_viewIndx,param,N,cubeD):
    """
    in order to save storage, we didn't store the maskID matrix {N,NO_viewPair*2,1,D,D,D} ( the NO_viewPair is very large).
    we want to use the viewPair_viewIndx and the param for each cube to generate this maskID matrix.
    inputs:
    viewPair_viewIndx[i]: each element {NO_viewPair, 2} is viewPair_viewIndx for a cube. The NO_viewPair may be different for different cubes because of visibility
    param[i]: each element is the cube params including: x,y,z,resol,modelIndx...
    return: 
    maskID_np: {N,NO_viewPair*2,1,cubeD,cubeD,cubeD}
    """
    NO_viewPair = viewPair_viewIndx.shape[1]
    maskID_list = []
    
    indx_xyz = range(0,cubeD)
    ##meshgrid: indexing : {'xy', 'ij'}, optional     ##Cartesian ('xy', default) or matrix ('ij') indexing of output.    
    indx_x,indx_y,indx_z = np.meshgrid(indx_xyz,indx_xyz,indx_xyz,indexing='ij')  

    for _viewPair_viewIndx, _param in zip(viewPair_viewIndx, param):
        min_x,min_y,min_z,resol,modelIndx = _param[:5]
        indx_x = indx_x * resol + min_x
        indx_y = indx_y * resol + min_y
        indx_z = indx_z * resol + min_z   
        homogen_1s = np.ones(cubeD**3, dtype=np.float64)
        pts_4D = np.vstack([indx_x.flatten(),indx_y.flatten(),indx_z.flatten(),homogen_1s])    
        
        maskID_list.append(generate_maskIDCubes_given_viewIndx(cameraPOs, cubeD, pts_4D, view_viewIndx=_viewPair_viewIndx.flatten()))
    return np.concatenate(maskID_list,axis=0)



#*********************
def pool_along_maskID(a_mask_value, initial, mask, value):
    """
    in each iteration:
    process all the 2D values having the same maskID
    a_mask_value: scalar in arange(NO_of_maskIDs+1)
    mask/value/return: {1,D,D}
    
    after scan/loop:
    scan's result: {NO_of_maskIDs, 1,D,D}
    """
    # impliment: a_location = T.argmax(value[mask==a_mask_value]).
    # BUT, the value[selection] and the == don't work as expected
    # replaced by T.where(?,a,b), and T.eq(a,b)
    ##"Cannot cast True or False as a tensor variable. Please use 1 or "
                ##"0. This error might be caused by using the == operator on "
                ##"Variables. v == w does not do what you think it does, "
                ##"use theano.tensor.eq(v, w) instead."    
    selection = T.eq(mask, a_mask_value) ## 0/1 ?

    masked_value = selection*value ##T.where(selection,value,0) / selection*value ## Attention: the min value of vriable 'value' should be larger than 0.
    a_maxValue = T.max(masked_value)
    ##There is 2 behavior of numpy.where(condition, [x ,y]). Theano always support you provide 3 parameter to where(). \
    ##As said in NumPy doc[1], numpy.where(cond) is equivalent to nonzero(). To get all the nonzero indices returned can use the nonzero() function~
    a_maxLocation = T.eq(masked_value, a_maxValue).nonzero()
    a_maxLocation = T.and_(selection, T.eq(masked_value, a_maxValue)).nonzero()
    
    return T.set_subtensor(initial[a_maxLocation],a_maxValue)

def process_a_mask(a_sample_mask, initial, a_sample_value):  
    """
    in each iteration:
    process each 2D value according to ONE 2D mask
    a_sample_mask/a_sample_value: {1,D,D}
    return: {1,D,D}
    
    after scan/loop:
    scan's result: {1,D,D}
    """    
    max_mask_value = T.max(a_sample_mask) 
    
    ##initial = T.zeros_like(a_sample_value)
    
    # very important note: the scan's result will automatically concatenate the results \
    # in each loop along the axis=0 of the variable 'result' into a new variable tensor
    result, updates = theano.scan(fn=pool_along_maskID,
                            sequences=[T.arange(max_mask_value + 1)],
                            outputs_info=initial,
                            non_sequences=[a_sample_mask, a_sample_value])
    return result[-1]##T.sum(result,axis=0) ##{NO_of_maskIDs, D,D} --> {1,D,D}



def process_a_sample(sample_masks, a_sample_value):  
    """
    in each iteration:
    process each 2D value according to M 2D masks, 
    sample_masks: {M,D,D}
    a_sample_value: {1,D,D}
    return: {1,D,D}
    
    after scan/loop:
    scan's result: {N,1,D,D}
    """    
    initial = T.zeros_like(a_sample_value)
    
    # very important note: the scan's result will automatically concatenate the results \
    # in each loop along the axis=0 of the variable 'result' into a new variable tensor
    result, updates = theano.scan(fn=process_a_mask,
                            sequences=[sample_masks],
                            outputs_info=initial,
                            non_sequences=[a_sample_value])
    return result[-1]##T.sum(result,axis=0) ##{NO_of_maskIDs, D,D} --> {1,D,D}


#####################################
# sequences input: {N,M,D,D} {N,1,D,D}
# output 'result': {N,1,D,D}
if define_theano_pool_ON:
    result, updates = theano.scan(fn=process_a_sample,
                          outputs_info=None,
                          sequences=[tensor_mask, tensor_value],
                          non_sequences=None)
    pool_along_mask_theano_fn = theano.function(inputs=[tensor_mask, tensor_value], outputs=[result])



#*********************
def maskID_pool_numpy(maskID_np, value_np, with_small_maskIDs = True):
    """ Do the max pooling among the values with the same maskID using Numpy.

    Inputs:
    ------------
    maskID_np: any shape, uint
            uint values are maskIDs 
    value_np: maskID_np.shape

    Outputs:
    ------------
    pooling_bool: maskID_np.shape
            True: for the position of max value among the same maskID

    Usage:
    ------------
    >>> maskID = np.array([[2,3],[1,2]])
    >>> value = np.array([[.2,30],[11,-1]])
    >>> pooling_bool = np.array([[1,1],[1,0]]).astype(np.bool)
    >>> np.allclose(maskID_pool_numpy(maskID, value, with_small_maskIDs = True), pooling_bool)
    True
    >>> np.allclose(maskID_pool_numpy(maskID, value, with_small_maskIDs = False), pooling_bool)
    True
    """
    pooling_bool = np.zeros_like(maskID_np).astype(np.bool)
    if with_small_maskIDs:
        maskID_max = maskID_np.max()
        mask_id_range = range(maskID_max+1)
    else:
        # because the np.unique takes long time
        mask_id_range = np.unique(maskID_np)
    # loop through all the maskIDs
    for _mask_id in mask_id_range:
        # tuple of ndim indexes for the positions with the same _mask_id
        indx_ndim = np.where(maskID_np == _mask_id)
        if indx_ndim[0].size == 0:
            continue # if there is no elements with _mask_id 
        _argmax_indx = np.argmax(value_np[indx_ndim])
        pooling_bool[[[_indx_1dim[_argmax_indx]] for _indx_1dim in indx_ndim]] = True
    return pooling_bool

def rayID_pooling_numpy(rayIDs, values):  
    """ Do the rayID pooling in Numpy.

    inputs:
    -------------
    rayIDs: (N_cube,N_view,D1,...), uint
            values are rayIDs in corresponding views.
    values: (N_cube,1,D1,...), float

    outputs:
    -------------
    pooling_bool: values.shape, uint

    Usage:
    -------------
    >>> values = np.reshape(np.arange(8), (2,1,2,2))
    >>> np.random.seed(201611)
    >>> rayIDs = np.random.randint(0,3,(2,3,2,2))
    >>> pooling_bool = np.array([[[[1,1],[3,3]]],[[[2,2],[2,3]]]])
    >>> np.allclose(rayID_pooling_numpy(rayIDs, values), pooling_bool)
    True
    """
    N_cube,N_view = rayIDs.shape[:2]
    N_max_votes_np = np.zeros_like(values).astype(np.uint16) # count how may max_ray_pooling votes for each voxel
    # loop for each cube
    for _cube in range(N_cube):
        value_np = values[_cube,0] # (N_cube,1,D1,...) ==> (D1,...)
        # loop for each viewIndx
        for _view in range(N_view):
            maskID_np = rayIDs[_cube, _view] # (N_cube,N_view,D1,...) ==> (D1,...)
            N_max_votes_np[_cube,0] += maskID_pool_numpy(maskID_np, value_np)
    return N_max_votes_np


#*********************
def process_each_maskID_numpy(sample_masks, sample_values, use_npMaskArgmax=False):  
    """ Do the maskID pooling in Numpy.
            Loop through each mask_id only.

    inputs:
    -------------
    sample_masks: (N_cube,N_view,D1,...), uint
            values are rayIDs in corresponding views.
    sample_values: (N_cube,1,D1,...), float

    outputs:
    -------------
    pooling_bool: values.shape, uint

    Usage:
    -------------
    >>> values = np.reshape(np.arange(8), (2,1,2,2))
    >>> np.random.seed(201611)
    >>> rayIDs = np.random.randint(0,3,(2,3,2,2))
    >>> pooling_bool = np.array([[[[1,1],[3,3]]],[[[2,2],[2,3]]]])
    >>> np.allclose(process_each_maskID_numpy(rayIDs, values, use_npMaskArgmax = False), pooling_bool)
    True
    >>> np.allclose(process_each_maskID_numpy(rayIDs, values, use_npMaskArgmax = True), pooling_bool)
    True
    """
    print_time_ON = False
    N_cube,N_view = sample_masks.shape[:2]
    sample_masks_flat, sample_values_flat_repeat = sample_masks.reshape(N_cube * N_view,-1), sample_values.repeat(N_view,axis=1).reshape(N_cube*N_view,-1) ## {N_cube*N_view,D**3}, {N_cube*N_view,D**3}
    maskID_max = sample_masks.max()
    
    argmax_label = np.zeros_like(sample_values_flat_repeat).astype(np.bool) # (N_cube*N_view, D**3)
    tmp_indices = np.arange(argmax_label.shape[0]) ## (N_cube*N_view,)

    for a_mask_value in range(maskID_max+1):
        select_a_mask_value = sample_masks_flat == a_mask_value ## {N_cube*N_view, D**3}
        select_valid_mask = select_a_mask_value.any(axis=-1)    ## bool{N_cube*N_view} because there may be no this perticular maskID in this mask,
        if use_npMaskArgmax:
            masked_values_flat_repeat_masked = np.ma.array(sample_values_flat_repeat, mask= ~select_a_mask_value) # ATTENTION, the element will be masked/ignored if mask=True
            masked_values_flat_argmax = np.ma.argmax(masked_values_flat_repeat_masked,axis=-1,fill_value=0)
        else:
            masked_values_flat = sample_values_flat_repeat * select_a_mask_value ## {N_cube*N_view, D**3}  # takes 40% time
            masked_values_flat_argmax = np.argmax(masked_values_flat,axis=-1) ## takes 40% time
        masked_values_flat_max_indices = (tmp_indices[select_valid_mask], masked_values_flat_argmax[select_valid_mask]) ## ([N_cube*N_view,],[N_cube*N_view,])
        argmax_label[masked_values_flat_max_indices] = True

    N_argmax_votes = argmax_label.reshape(sample_masks.shape).sum(axis=1, keepdims=True)
    return N_argmax_votes


#*********************
def process_each_maskID_npSort(sample_masks, sample_values, onlyReturn_mask=False):  
    """
    Do the maskID pooling using sorted Numpy.
    
    sample_masks {N,M,D,D,D}
    sample_values {N,1,D,D,D}
    """
    if onlyReturn_mask:
        NO_of_pooled_values = np.zeros_like(sample_values).astype(np.uint16)
    else:
        pooled_values = np.zeros_like(sample_values) #not necessary
    
    N,M,D,_,_ = sample_masks.shape
    ind_N, ind_M, ind_D1, ind_D2, ind_D3 = np.indices((N,M,D,D,D)).reshape((5,-1))
    masks = sample_masks.flatten()
    values = sample_values.repeat(M,axis=1).flatten()
    sortedIndx_NMmv = np.lexsort((values,masks,ind_M,ind_N))[::-1] # because the last one is the maximum
    ## after this func, the output.shape[0]-x.shape[0]=1 (the first element)
    valueChangeBool = lambda x: (x[sortedIndx_NMmv][1:]-x[sortedIndx_NMmv][:-1]).astype(np.bool) 
    ## select the first indx in the sorted group, in which all the masks,ind_M,ind_N values are equal.
    selected_SortedIndx = sortedIndx_NMmv[1:][valueChangeBool(ind_N) | valueChangeBool(ind_M) | valueChangeBool(masks)]
    ## because the first indx is valid, should be stacked into the selected_SortedIndx
    selected_SortedIndx = np.hstack([sortedIndx_NMmv[0],selected_SortedIndx])
    selected_sampleIndx = [x[selected_SortedIndx] for x in [ind_N, ind_M, ind_D1, ind_D2, ind_D3]]
    selected_sampleIndx[1] = np.zeros(selected_SortedIndx.shape).astype(np.int) #
    if onlyReturn_mask:
        NO_of_pooled_values[selected_sampleIndx] += 1 # BUT max(NO_of_pooled_values_flat) = 1
        return NO_of_pooled_values
    else:    
        pooled_values[selected_sampleIndx] = sample_values[selected_sampleIndx]
        return pooled_values
    

#*********************
def process_each_maskID_gnumpy(sample_masks, sample_values, onlyReturn_mask=False):  
    """
    Do the maskID pooling in Numpy.
    sample_masks {N,M,D,D,D}
    sample_values {N,1,D,D,D}
    """
    import gnumpy as gpu
    sample_masks, sample_values = gpu.garray(sample_masks).astype('uint32'), gpu.garray(sample_values)
    
    N,M,D,_,_ = sample_masks.shape
    sample_masks_flat, sample_values_flat = sample_masks.reshape(N,M,-1), sample_values.reshape(N,1,-1) ## {N,M,D**3}, {N,1,D**3}
    maskID_max = int(sample_masks.max())
    
    if onlyReturn_mask:
        NO_of_pooled_values_flat = gpu.garray(np.zeros(sample_values_flat.shape).astype(np.uint16))
    else:
        pooled_values_flat = gpu.garray(np.zeros(sample_values_flat.shape))
    
    for a_mask_value in range(maskID_max+1):
        select_a_mask_value = sample_masks_flat == a_mask_value ## {N,M,D**3}
        select_valid_mask = select_a_mask_value.any(axis=-1)    ## bool{N,M} because there may be no this perticular maskID in this mask,
        masked_values_flat = sample_values_flat * select_a_mask_value ## {N,1,D**3}*{N,M,D**3} = {N,M,D**3}  # takes 40% time
        masked_values_flat_argmax = np.argmax(masked_values_flat.as_numpy_array(),axis=-1) ## takes 40% time
        tmp_indices = np.indices((N,M)) ## {2,N,M}
        tmp_indices[1] = 0 # no matter which mask (axis=1) determines to keep each max value, we keep them all.
        masked_values_flat_max_indices = tuple(tmp_indices[:,select_valid_mask]) + (masked_values_flat_argmax[select_valid_mask],) ## ([N,M],[N,M],[N,M]) --> ([N',],[N',],[N',])
        if onlyReturn_mask:
            NO_of_pooled_values_flat[masked_values_flat_max_indices] += 1
        else:
            pooled_values_flat[masked_values_flat_max_indices] = sample_values_flat[masked_values_flat_max_indices]
    
    if onlyReturn_mask:
        return NO_of_pooled_values_flat.reshape(sample_values.shape)
    else:
        return pooled_values_flat.reshape(sample_values.shape)



#*********************
from multiprocessing import Pool
from functools import partial
from contextlib import contextmanager

@contextmanager
def terminating(thing): # thanks: http://stackoverflow.com/questions/25968518/python-multiprocessing-lib-error-attributeerror-exit
    try:
        yield thing
    finally:
        thing.terminate()
def func(sample_masks_flat, sample_values_flat, a_mask_value):
    select_a_mask_value = sample_masks_flat == a_mask_value ## {N,M,D**3}
    select_valid_mask = select_a_mask_value.any(axis=-1)    ## bool{N,M} because there may be no this perticular maskID in this mask,
    NO_of_pooled_values_flat = np.zeros_like(sample_values_flat).astype(np.uint16)   # (N,1,-1)
    masked_values_flat = sample_values_flat * select_a_mask_value ## {N,M,D**3}
    masked_values_flat_argmax = np.argmax(masked_values_flat,axis=-1)
    tmp_indices = np.indices(sample_masks_flat.shape[:2]) #(N,M)) ## {2,N,M}
    tmp_indices[1] = 0 # no matter which mask (axis=1) determines to keep each max value, we keep them all.
    masked_values_flat_max_indices = tuple(tmp_indices[:,select_valid_mask]) + (masked_values_flat_argmax[select_valid_mask],) ## ([N,M],[N,M],[N,M]) --> ([N',],[N',],[N',])
    masked_values_flat_max_indices_np = np.c_[masked_values_flat_max_indices]
    for position in masked_values_flat_max_indices_np:
        NO_of_pooled_values_flat[tuple(position)] += 1
    return NO_of_pooled_values_flat
        
def multiprocess_samples(sample_masks, sample_values, NO_of_processes=3, with_small_maskIDs = True):
    """
    Usage:
    ------------
    >>> values = np.reshape(np.arange(8), (2,1,2,2))
    >>> np.random.seed(201611)
    >>> rayIDs = np.random.randint(0,3,(2,3,2,2))
    >>> pooling_bool = np.array([[[[1,1],[3,3]]],[[[2,2],[2,3]]]])
    >>> np.allclose(multiprocess_samples(rayIDs, values), pooling_bool)
    True
    """
    ##sample_masks, sample_values = np_mask,np_value
    N,M = sample_masks.shape[:2]
    sample_masks_flat, sample_values_flat = sample_masks.reshape(N,M,-1), sample_values.reshape(N,1,-1) ## {N,M,D**3}, {N,1,D**3}
    if with_small_maskIDs:
        maskID_max = sample_masks.max()
        mask_id_range = range(maskID_max+1)
    else:
        # because the np.unique takes long time
        mask_id_range = np.unique(sample_masks)
    
    f = partial(func, sample_masks_flat, sample_values_flat) # thanks: http://stackoverflow.com/questions/24728084/why-does-this-implementation-of-multiprocessing-pool-not-work
    
    with terminating(Pool(NO_of_processes)) as pool:
        outputs = pool.map(f, mask_id_range)
    return np.stack(outputs).sum(axis=0).reshape(sample_values.shape).astype(np.uint8)



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Test the function: 
#pool_along_mask_fn
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


if debug_ON:
    
    # np_value = np.reshape(np.arange(8), (2,1,2,2))
    # np.random.seed(201611)
    # np_mask = np.random.randint(0,3,(2,3,2,2))
    np_mask=np.random.randint(0,26,[1*const,2*const**2,3*const,3*const,3*const]).astype(np.uint16)##(np.random.random((2,5,5))*26).astype(np.uint8)
    np_value=np.random.random([1*const,1,3*const,3*const,3*const]).astype(np.float32)# float16 is not as efficient as float32 ##(np.random.random(np_mask.shape)*26).astype(np.uint8)
    print "scan_RayTracing.py DEBUG MODE: attention: the numpy operation on float16 is not as efficient as float32. at least 20X slower"
    
    if print_output_ON: print np_mask, '\n', np_value
    
    if define_theano_pool_ON:
        t = time.time()
        output1 = pool_along_mask_theano_fn(np_mask,np_value) ## 26,[3,2/1,50,50,50] --> 0.33 s
        if print_output_ON: print output1 #np.asarray(output).sum(axis=0)
        print "\n rayTrace.pool_along_mask_theano_fn, time:  ", time.time()-t
    
    if define_numpy_pool_ON:
        t = time.time()
        output2 = process_each_maskID_numpy(np_mask,np_value,use_npMaskArgmax=False)
        if print_output_ON: print '\n', output2
        print "\n rayTrace.process_each_maskID_numpy, time:  ", time.time()-t #np.asarray(output).sum(axis=0)
        # t = time.time()
        # output = process_each_maskID_numpy(np_mask,np_value,use_npMaskArgmax=True)
        # if print_output_ON: print '\n', output2
        # print "\n rayTrace.process_each_maskID_numpy_MASK, time:  ", time.time()-t #np.asarray(output).sum(axis=0)        
        # print (output!=output2).sum()

        # t = time.time()
        # output3 = rayID_pooling_numpy(np_mask,np_value)
        # if print_output_ON: print '\n', output2
        # print "\n rayTrace.rayID_pooling_numpy, time:  ", time.time()-t #np.asarray(output).sum(axis=0)
        # print (output!=output3).sum()
        # print (output2!=output3).sum()
    
    if define_npSort_pool_ON:
        t = time.time()
        output = process_each_maskID_npSort(np_mask,np_value)
        if print_output_ON: print '\n', output
        print "\n rayTrace.process_each_maskID_npSort, time:  ", time.time()-t #np.asarray(output).sum(axis=0)
            
    if define_process_each_maskID_gnumpy_ON:
        t = time.time()
        output_gnumpy = process_each_maskID_gnumpy(np_mask,np_value)
        if print_output_ON: print '\n', output_gnumpy
        print "\n rayTrace.process_each_maskID_gnumpy, time:  ", time.time()-t #np.asarray(output).sum(axis=0)
    
    if define_multiprocess_pool_ON:
        t = time.time()
        output3 = multiprocess_samples(np_mask,np_value, NO_of_processes=6)
        if print_output_ON: print '\n', output3
        print "\n rayTrace.multiprocess_samples, time:  ", time.time()-t #np.asarray(output).sum(axis=0)

        print np.allclose(output2, output3)
    ##print 'done'
    ##return pooled_values_flat.reshape(sample_values.shape)
    

import doctest
doctest.testmod()
