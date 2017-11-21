
import numpy as np

def generate_voxelLevelWeighted_coloredCubes(viewPair_coloredCubes, viewPair_surf_predictions, weight4viewPair):
    """
    fuse the color based on the viewPair's colored cubes, surface predictions, and weight4viewPair

    inputs
    -----
    weight4viewPair (N_cubes, N_viewPairs): relative importance of each viewPair
    viewPair_surf_predictions (N_cubes, N_viewPairs, D,D,D): relative importance of each voxel in the same cube
    viewPair_coloredCubes (N_cubes * N_viewPairs, 6, D,D,D): rgb values from the views in the same viewPair 
        randomly select one viewPair_coloredCubes (N_cubes, N_viewPairs, 3, D,D,D), otherwise the finnal colorized cube could have up/down view bias
        or simply take average

    outputs
    ------
    new_coloredCubes: (N_cubes, 3, D,D,D)

    notes
    ------
    The fusion idea is like this: 
        weight4viewPair * viewPair_surf_predictions = voxel_weight (N_cubes, N_viewPairs, D,D,D) generate relative importance of voxels in all the viewPairs
        weighted_sum(randSelect_coloredCubes, normalized_voxel_weight) = new_colored_cubes (N_cubes, 3, D,D,D)
    """
    N_cubes, N_viewPairs, _D = viewPair_surf_predictions.shape[:3]
    # (N_cubes, N_viewPairs,1,1,1) * (N_cubes, N_viewPairs, D,D,D) ==> (N_cubes, N_viewPairs, D,D,D)
    voxel_weight = weight4viewPair[...,None,None,None] * viewPair_surf_predictions
    voxel_weight /= np.sum(voxel_weight, axis=1, keepdims=True) # normalization along different view pairs

    # take average of the view0/1
    # (N_cubes, N_viewPairs, 2, 3, D,D,D) ==> (N_cubes, N_viewPairs, 3, D,D,D) 
    mean_viewPair_coloredCubes = np.mean(viewPair_coloredCubes.astype(np.float32).reshape((N_cubes, N_viewPairs, 2, 3, _D,_D,_D)), axis=2)

    # sum[(N_cubes, N_viewPairs, 1, D,D,D) * (N_cubes, N_viewPairs, 3, D,D,D), axis=1] ==>(N_cubes, 3, D,D,D)
    new_coloredCubes = np.sum(voxel_weight[:,:,None,...] * mean_viewPair_coloredCubes, axis=1)

    return new_coloredCubes.astype(np.uint8)


def gen_batch_index(N_all, batch_size):
    """
    return list of index lists, which can be used to access each batch

    ---------------
    inputs:
        N_all: # of all elements
        batch_size: # of elements in each batch
    outputs:
        batch_index_list[i] is the indexes of batch i.
    
    ---------------
    notes:
        Python don't have out range check, the simpliest version could be:
        for _i in range(0, N_all, batch_size):
            yield range(_i, _i + batch_size)
    ---------------
    examples:
    >>> gen_batch_index(6,3) == [[0,1,2],[3,4,5]]
    True
    >>> gen_batch_index(7,3) == [[0,1,2],[3,4,5],[6]]
    True
    >>> gen_batch_index(8,3) == [[0,1,2],[3,4,5],[6,7]]
    True
    """
    batch_index_list = []
    for _batch_start_indx in range(0, N_all, batch_size):
        _batch_end_indx = min(_batch_start_indx + batch_size, N_all)
        batch_index_list.append(range(_batch_start_indx, _batch_end_indx))
    return batch_index_list


import doctest
doctest.testmod()

