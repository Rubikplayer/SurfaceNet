import numpy as np
import os
from PIL import Image


def load_cameraT_as_np(cameraT_folder):
    """
    load cameraT files as numpy

    Parameters
    ----------
    cameraT_folder: the folder containing the cameraT files

    Returns
    -------
    cameraT_array: cameraT_array[i] is corresponding to the ith camera
    """
    cameraT_files = os.listdir(cameraT_folder)
    NO_views_possible = len(cameraT_files)
    cameraT_array = np.zeros((NO_views_possible+1,3),dtype=np.float32) # indx_view starts from 1
    for _cameraT_file in cameraT_files:
        indx_view = int(_cameraT_file.split(".")[0].split("T")[-1])
        with open(cameraT_folder+_cameraT_file) as _cameraT_f:
            cameraT_array[indx_view] = np.asarray(_cameraT_f.read().split("\n")[:3]).astype(np.float32)
    return cameraT_array

def cameraP2T(cameraPO):
    """
    cameraPO: (3,4)
    return camera center in the world coords: cameraT (3,0)
    >>> P = np.array([[798.693916, -2438.153488, 1568.674338, -542599.034996], \
                  [-44.838945, 1433.912029, 2576.399630, -1176685.647358], \
                  [-0.840873, -0.344537, 0.417405, 382.793511]])
    >>> t = np.array([555.64348632032, 191.10837560939, 360.02470478273])
    >>> np.allclose(cameraP2T(P), t)
    True
    """
    homo4D = np.array([np.linalg.det(cameraPO[:,[1,2,3]]), -1*np.linalg.det(cameraPO[:,[0,2,3]]), np.linalg.det(cameraPO[:,[0,1,3]]), -1*np.linalg.det(cameraPO[:,[0,1,2]]) ])
    cameraT = homo4D[:3] / homo4D[3]
    return cameraT

    
def cameraPs2Ts(cameraPOs):
    """

    """
    if type(cameraPOs) is list:
        N = len(cameraPOs)
    else:                
        N = cameraPOs.shape[0]
    cameraT_list = []    
    for _cameraPO in cameraPOs:
        cameraT_list.append(cameraP2T(_cameraPO))

    return cameraT_list if type(cameraPOs) is list else np.stack(cameraT_list)



def load_cameraPos_as_np(cameraPos_folder):
    """
    load cameraPos files as numpy

    Parameters
    ----------
    cameraPos_folder: the folder containing the cameraT files

    Returns
    -------
    cameraPos_array: cameraPos_array[i] is corresponding to the ith camera
    """
    cameraPos_files = os.listdir(cameraPos_folder)
    NO_views_possible = len(cameraPos_files)
    cameraPos_array = np.zeros((NO_views_possible+1,3,4),dtype=np.float32) # indx_view starts from 1
    for _cameraPos_file in cameraPos_files:
        indx_view = int(_cameraPos_file.split('_')[1].split('.')[0])
        cameraPos_array[indx_view] = np.loadtxt(cameraPos_folder+_cameraPos_file)
    return cameraPos_array


def perspectiveProj(projection_M, xyz_3D, return_int_wh = True, return_depth = False):
    """ perform perspective projection to vectors given projection matrixes
        support multiple projection_matrixes and multiple 3D vectors
        notice: [matlabx,matlaby] = [width, height]

    Parameters
    ----------
    projection_M: numpy with shape (3,4) / (N_Ms, 3,4)
    xyz_3D: numpy with shape (3,) / (N_pts, 3)
    return_int_wh: bool, round results to integer when True.

    Returns
    -------
    img_w, img_h: (N_pts,) / (N_Ms, N_pts)

    Examples
    --------

    inputs: (N_Ms, 3,4) & (N_pts, 3), return_int_wh = False/True

    >>> np.random.seed(201611)
    >>> Ms = np.random.rand(2,3,4)
    >>> pts_3D = np.random.rand(2,3)
    >>> pts_2Dw, pts_2Dh = perspectiveProj(Ms, pts_3D, return_int_wh = False)
    >>> np.allclose(pts_2Dw, np.array([[ 1.35860185,  0.9878389 ],
    ...        [ 0.64522543,  0.76079278 ]]))
    True
    >>> pts_2Dw_int, pts_2Dh_int = perspectiveProj(Ms, pts_3D, return_int_wh = True)
    >>> np.allclose(pts_2Dw_int, np.array([[1, 1], [1, 1]]))
    True

    inputs: (3,4) & (3,)

    >>> np.allclose(
    ...         np.r_[perspectiveProj(Ms[1], pts_3D[0], return_int_wh = False)],
    ...         np.stack((pts_2Dw, pts_2Dh))[:,1,0])
    True
    """
    if projection_M.shape[-2:] != (3,4):
        raise ValueError("perspectiveProj needs projection_M with shape (3,4), however got {}".format(projection_M.shape))

    if xyz_3D.ndim == 1:
        xyz_3D = xyz_3D[None,:]

    if xyz_3D.shape[1] != 3 or xyz_3D.ndim != 2:
        raise ValueError("perspectiveProj needs xyz_3D with shape (3,) or (N_pts, 3), however got {}".format(xyz_3D.shape))
    # perspective projection
    N_pts = xyz_3D.shape[0]
    xyz1 = np.c_[xyz_3D, np.ones((N_pts,1))].astype(np.float64) # (N_pts, 3) ==> (N_pts, 4)
    pts_3D = np.matmul(projection_M, xyz1.T) # (3, 4)/(N_Ms, 3, 4) * (4, N_pts) ==> (3, N_pts)/(N_Ms,3,N_pts)
    # the result is vector: [w,h,1], w is the first dim!!! (matlab's x/y/1')
    pts_2D = pts_3D[...,:2,:]
    pts_2D /= pts_3D[...,2:3,:] # (2, N_pts) /= (1, N_pts) | (N_Ms, 2, N_pts) /= (N_Ms, 1, N_pts)
    if return_int_wh: 
        pts_2D = pts_2D.round().astype(np.int64)  # (2, N_pts) / (N_Ms, 2, N_pts)
    img_w, img_h = pts_2D[...,0,:], pts_2D[...,1,:] # (N_pts,) / (N_Ms, N_pts)
    if return_depth:
        depth = pts_3D[...,2,:]
        return img_w, img_h, depth
    return img_w, img_h



def crop_imgPatch_of_3Dpt(img, projection_M, xyz_3D, patch_r, return_PIL_or_np="np"):
    """
    generate patch of 3D points [x,y,z] given the projection matrix and image 
    returned patch could include black

    Parameters
    ----------
    projection_M: camera projection matrix, 
        numpy with shape (3,4)
    img: numpy with shape (h,w,c), where c should < 4. because using PIL lib to load numpy
    xyz_3D: numpy with shape (3,) / (N_pts, 3)
    patch_r: half of the patch width/height
    return_PIL_or_np: if this func is followed by some preprocessing, return PIL could save time

    Returns
    -------
    returned_patches: if only one xyz_3D is given, return Image/numpy with shape (2*patch_r,2*patch_r,c), where c is color channel
            if xyz_3D.ndim = 2, return list of Image/numpy with ...
    patchCenter_inScope: return True if the patch center is in scope of the image.

    Examples
    --------
    """
    if img.ndim != 3:
        raise ValueError("crop_imgPatch_of_3Dpt needs img with 3 ndim, however got {}".format(img.ndim))
    
    if xyz_3D.ndim == 1:
        _return_only_one_patch = True
        xyz_3D = xyz_3D[None,:]

    if xyz_3D.shape[1] != 3 or xyz_3D.ndim != 2:
        raise ValueError("perspectiveProj needs xyz_3D with shape (3,) or (N_pts, 3), however got {}".format(xyz_3D.shape))

    N_pts = xyz_3D.shape[0]
    proj_w, proj_h = perspectiveProj(projection_M, xyz_3D)

    # because we want to use img[[[2,2],[3,3]], [[7,8],[7,8]]] to access the pixels with shape like:
    # [pixel[2,7],pixel[2,8]
    #  pixel[3,7],pixel[3,8]]
    # the # should not be out of range of the array.shape
    # inScope patch centers would be that the boundary of the patch also within the range 
    # (unlike PIL, which just leave the outside pixel as 0, but takes long time) 
    img_h, img_w = img.shape[:2]
    patchCenter_inScope = (proj_w >= patch_r) & (proj_w <= (img_w - patch_r)) & (proj_h >= patch_r) & (proj_h <= (img_h - patch_r)) # (N_pts,)
    # if patch_r = 2, generate the array like:
    # [[-2,-2,-2,-2],...,...,[1,1,1,1]] and [[-2,-1,0,1],...,[-2,-1,0,1]]
    # with shape (2, 2*patch_r, 2*patch_r)
    # so that after adding the center shift coordinates, we can get the indices of the pixels of the patch in the img
    patchPixel_coord_wrt_center = np.indices((patch_r * 2,patch_r * 2)) - patch_r
    patchCenter_coord = np.c_[proj_h, proj_w] # [N_pts, 2]
    # for the out scope patch centers, we just assign some safe number to it, like patch_r. Latter on filter out those patches
    patchCenter_coord[~patchCenter_inScope] = patch_r
    # (N_pts, 2, 1, 1) + (1, 2, 2*patch_r, 2*patch_r) ==> (N_pts, 2, 2*patch_r, 2*patch_r)
    patchPixel_coord = patchCenter_coord[...,None,None] + patchPixel_coord_wrt_center[None,...]
    patches_of_allPts = img[patchPixel_coord[:,0], patchPixel_coord[:,1], :] # (N_pts, 2*patch_r, 2*patch_r, c)
    # set the outscope patches to all0
    patches_of_allPts[~patchCenter_inScope] = 0 
    returned_patches = patches_of_allPts # (N_pts, 2*patch_r, 2*patch_r, c)

    return returned_patches, patchCenter_inScope


def calculate_angle_p1_p2_p3(p1,p2,p3,return_angle=True, return_cosAngle=True):
    """
    calculate angle <p1,p2,p3>, which is the angle between the vectors p2p1 and p2p3 

    Parameters
    ----------
    p1/p2/p3: numpy with shape (3,)
    return_angle: return the radian angle
    return_cosAngle: return the cosine value

    Returns
    -------
    angle, cosAngle

    Examples
    --------
    """
    unit_vector = lambda v: v / np.linalg.norm(v)
    angle = lambda v1,v2: np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0))
    cos_angle = lambda v1,v2: np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0)

    vect_p2p1 = p1-p2
    vect_p2p3 = p3-p2
    return angle(vect_p2p1, vect_p2p3) if return_angle else None , \
            cos_angle(vect_p2p1, vect_p2p3) if return_cosAngle else None



import doctest
doctest.testmod()
