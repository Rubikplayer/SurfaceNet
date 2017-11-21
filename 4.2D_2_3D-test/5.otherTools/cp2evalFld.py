import thread
import os
import sys

"""
This piece of code tries to copy the 
xx/adapt_thresh/model{n}-6viewPairs-resol0.400-strideRatio0.500/xx/adapThresh_rayPool_Off-init2.0_Ndecision0_iter{m}_realColor.ply
to the matlab evaluation folder:
/home/mengqi/dataset/MVS/SampleSet/MVS Data/Points/mji-iter{m}-/mji-iter{m}-{:03d}_l3.ply


usage
--------
python pcd2ply n
"""

modelIndx = int(sys.argv[1])
adapt_thresh_fld = "/home/mengqi/dataset/MVS/lasagne/save_reconstruction_result/adapt_thresh/model{}-6viewPairs-resol0.400-strideRatio0.500/".format(modelIndx)
matlab_eval_fld = "/home/mengqi/dataset/MVS/SampleSet/MVS\ Data/Points"
for root, dirs, files in os.walk(adapt_thresh_fld):
    if root == adapt_thresh_fld:
        if len(dirs) != 1:
            print("warning: there are multiple sub-directories in the folder: {}".format(root))
            break
    else: # when the 'root' goes into subdirectory
        for _file in files:
            if _file.endswith('_realColor.ply'):
                iter = _file.split('_')[-2][4:] # read the _iter3_xxx number
                orig_file = os.path.join(root, _file)
                desti_file = os.path.join(matlab_eval_fld, 'mji-iter{0}-/mji-iter{0}-{1:03d}_l3.ply'.format(iter, modelIndx))
                os.system('cp {} {}'.format(orig_file, desti_file))
                print("copied to {}".format(desti_file))




