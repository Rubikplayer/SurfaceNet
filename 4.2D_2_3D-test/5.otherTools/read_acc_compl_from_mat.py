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
# python read_acc_compl_from_mat.py /media/mengqi/data/mengqi/dataset/MVS/SampleSet/MVS\ Data/Results
"""

modelIndx_range = range(1,129) # 129
method_names = ['Tola','Furu','Camp','gipuma']#  + ['mji-iter{}-'.format(i) for i in range(15)]
N_models = modelIndx_range[-1]
N_methods = len(method_names)
eval_results = np.zeros((N_methods, N_models, 4)) # 4 means: median/mean-acc/compl
with open('./acc_compl.txt', mode='w+') as f:
    mat_fld = sys.argv[-1]
    f.write("mean(BaseEval.Ddata .* BaseEval.DataInMask), median(BaseEval.Ddata .* BaseEval.DataInMask), mean(BaseEval.Dstl .* BaseEval.StlAbovePlane), median(BaseEval.Dstl .* BaseEval.StlAbovePlane) \n")
    for _modelIndx in modelIndx_range:
        f.write('model: {} \n'.format(_modelIndx))
        for _n_method, _method_name in enumerate(method_names):
            mat_file = '{}_Eval_IJCV_{}.mat'.format(_method_name, _modelIndx)
            mat_file_path = os.path.join(mat_fld, mat_file)
            acc_compl_str = _method_name + ': n/a \n'
            try:
                BaseEval = scipy.io.loadmat(mat_file_path)
                DataInMask = BaseEval['BaseEval'][0,0]['DataInMask']
                StlAbovePlane = BaseEval['BaseEval'][0,0]['StlAbovePlane']
                Ddata = BaseEval['BaseEval'][0,0]['Ddata']
                Dstl = BaseEval['BaseEval'][0,0]['Dstl']
                Ddata *= DataInMask
                Dstl *= StlAbovePlane
                eval_results[_n_method, _modelIndx] = [Ddata.mean(),np.median(Ddata),Dstl.mean(), np.median(Dstl)]
                acc_compl_str = '{}: {} \t{} \t{} \t{} \n'.format(_method_name, Ddata.mean(),np.median(Ddata),Dstl.mean(), np.median(Dstl))
            except:
                pass
            f.write(acc_compl_str)
    np.save('./acc_compl', eval_results)






