import os
import sys
import scipy.io
import numpy as np


"""
This piece of code read the MVS dataset's evaluation result .mat file.
Read the mean / median of accuracy / completeness of different models, save to file.

TODO: save the NO.s to .npz also.

usage
------
# python read_acc_compl_from_mat.py xx.mat
"""

mat_file_path = sys.argv[-1]
print(mat_file_path)
try:
    BaseEval = scipy.io.loadmat(mat_file_path)
    DataInMask = BaseEval['BaseEval'][0,0]['DataInMask']
    StlAbovePlane = BaseEval['BaseEval'][0,0]['StlAbovePlane']
    Ddata = BaseEval['BaseEval'][0,0]['Ddata']
    Dstl = BaseEval['BaseEval'][0,0]['Dstl']
    Ddata *= DataInMask
    Dstl *= StlAbovePlane
    print('{} \t{} \t{} \t{}'.format(Ddata.mean(),np.median(Ddata),Dstl.mean(), np.median(Dstl)))
except:
    print('except')






