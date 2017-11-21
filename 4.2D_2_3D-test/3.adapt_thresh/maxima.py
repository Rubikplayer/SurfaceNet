from scipy import signal as signal
import numpy as np

def localmax_ndim(data, axes=None):
    """
    mark local max along each axis

    Parameters:
    ----------
    data: ndim numpy array
    axes: axes along which the local maxima will be extracted. e.g.: [1,3]
        If `None`, axes = range(data.ndim)


    Returns:
    ----------
    localmax: localmax[i] stores local max along dimention axes[i].
        `True` means local max along the corresponding axis.

    Example:
    ----------
    >>> from maxima import localmax_ndim
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 6, 0],
    ...               [5, 3, 4, 4]])
    >>> output = np.array([[[False, False, False, False],                                                                    
    ...                     [False, False,  True, False],                                                                    
    ...                     [False, False, False, False]],                                                                   
    ...                    [[False,  True, False, False],                                                                    
    ...                     [False, False,  True, False],                                                                    
    ...                     [False, False, False, False]]], dtype=bool)
    >>> np.allclose(output, localmax_ndim(y, axes=[0,1]))
    True
    >>> np.allclose(output, localmax_ndim(y, axes=None))
    True
    """
    if axes == None:
        axes = range(data.ndim)
    localmax = np.zeros((len(axes),) + data.shape).astype(np.bool)
    for _n, _axis in enumerate(axes):
        indices = signal.argrelmax(data, axis=_axis, order=1, mode='clip')
        localmax[_n][indices] = True
    return localmax


def globmax_ndim(data, axes=None):
    """
    mark all global max positions along each axis

    - np.amax along an axis
    - broadcast along that axis
    - elementwise comparison with the orig data.

    Parameters:
    ----------
    data: ndim numpy array
    axes: axes along which the local maxima will be extracted. e.g.: [1,3]
        If `None`, axes = range(data.ndim)

    Returns:
    ----------
    globmax: globmax[i] stores global max along dimention axes[i].
        `True` means global max along the corresponding axis.

    Example:
    ----------
    >>> from maxima import globmax_ndim
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 6, 0],
    ...               [5, 3, 4, 4]])
    >>> output = np.array([[[False, False, False, False],                                                                    
    ...                     [False, False,  True, False],                                                                    
    ...                     [True,  True,  False, True]],                                                                   
    ...                    [[False, True, False, True],                                                                    
    ...                     [False, False,  True, False],                                                                    
    ...                     [True,  False, False, False]]], dtype=bool)
    >>> np.allclose(output, globmax_ndim(y, axes=None))
    True
    >>> np.allclose(output[::-1], globmax_ndim(y, axes=[-1,0]))
    True
    """
    if axes == None:
        axes = range(data.ndim)
    dshape = data.shape
    globmax = np.zeros((len(axes),) + dshape).astype(np.bool)
    for _n, _axis in enumerate(axes):
        globmax[_n] = data == \
                np.repeat(np.amax(data, axis=_axis, keepdims=True), dshape[_axis], axis=_axis)
    return globmax




import doctest
doctest.testmod()
