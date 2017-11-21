import numpy as np
import os

class sparseDict(dict):
    """ dict with default value
    
    """
    def __init__(self, default_value = None):
        dict.__init__(self)
        self._defaultValue = default_value

    def setDefaltValue(self, default_value):
        self._defaultValue = default_value

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return self._defaultValue
    def __setitem__(self, key, val):
        # I'm sure this can go faster if I were smarter
        val_is_defaultValue = False

        if type(val) == type(self._defaultValue):
            if isinstance(val, np.ndarray):
                if np.array_equal(val, self._defaultValue):
                    val_is_defaultValue = True
            elif val == self._defaultValue:
                val_is_defaultValue = True
        # else:
        #     print("Warning: the assigned value type {} != default type {}".format(type(val), type(self._defaultValue)))

        if val_is_defaultValue:
            if  key in self:
                del self[key]
        else:
            dict.__setitem__(self, key, val)

def load_numpy_sparseDict(file_name, with_printf = True):
    """ save the sparseDict to npz file
            only support the case: key=tuple(#,...,#); value=numpy array
    """
    if not os.path.isfile(file_name):
        raise Warning('load_numpy_sparseDict func cannot load file: {}'.format(file_name))
    with open(file_name) as f:
        npz_file = np.load(f)
        default_value_np, dict_keys_np, dict_values_np = npz_file['default_value_np'], npz_file['dict_keys_np'], npz_file['dict_values_np']
    sparse_dict = sparseDict(default_value_np)
    sparse_dict.update({tuple(_key):_value for _key, _value in zip(dict_keys_np, dict_values_np)})
    if with_printf:
        print("loaded sparse_dict from file: {}".format(file_name))
    return sparse_dict

def save_numpy_sparseDict(file_name, sparse_dict, with_printf = True):
    """ save the sparseDict to npz file
            only support the case: key=tuple(#,...,#); value=numpy array

    Usage:
    ----------------
    >>> sparse_dict = sparseDict(np.random.rand(3,5))
    >>> sparse_dict[1,2,3] = np.random.rand(3,5)
    >>> file_name = './_tmp_dict.npz'
    >>> save_numpy_sparseDict(file_name, sparse_dict, with_printf = False) # no printf in order to pass the doctest
    >>> sparse_dict2 = load_numpy_sparseDict(file_name, with_printf = False)
    >>> np.allclose(sparse_dict._defaultValue, sparse_dict2._defaultValue)
    True
    >>> np.allclose(np.array(sparse_dict.keys()), np.array(sparse_dict2.keys()))
    True
    >>> np.allclose(np.array(sparse_dict.values()), np.array(sparse_dict2.values()))
    True
    >>> os.remove(file_name)
    """
    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    default_value_np = sparse_dict._defaultValue # should be numpy
    dict_keys_np = np.array(sparse_dict.keys())
    dict_values_np = np.array(sparse_dict.values())

    with open(file_name, 'wb') as f:
        np.savez_compressed(f, default_value_np = default_value_np, dict_keys_np = dict_keys_np, dict_values_np = dict_values_np)
    if with_printf:
        print('saved sparse_dict into file: {}'.format(file_name))
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()
