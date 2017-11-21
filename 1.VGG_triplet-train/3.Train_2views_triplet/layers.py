""" 
define some customized layers 
only loaded by *Net.py files
"""

import lasagne

class Norm_L2_Layer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        input_sqr = input**2
        input_L2 = (input_sqr.sum(axis=1))**.5
        input_unit = input/input_L2[:,None]        
        return input_unit
    
class Get_center_2x2(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        center = self.input_shape[-1]/2
        input_center_2x2 = input[:,:,center-1:center+1, center-1:center+1]
        return input_center_2x2.flatten(2)
    def get_output_shape_for(self, input_shape):
        return [input_shape[0], input_shape[1]*2*2]

class Euclid_dist_Layer(lasagne.layers.MergeLayer):
    """
    compute euclidian distance between 2 layers along a specified dim.
    ||f1-f2||_2^2 if L2_square else ||f1-f2||_2

    Parameters
    ----------
    incoming:
        two layers with the same output_shape
    axis=1:
        treat which axis as a euclidian vector (the dim of the f1/f2 in (f1-f2)**.5)
    keepdims=True:
        whether keep the dim on which the euclid dist is performed
    """
    def __init__(self, incoming, axis=1, keepdims=True, L2_square = False, **kwargs):
        super(Euclid_dist_Layer, self).__init__(incoming, **kwargs)
        numInputs = len(self.input_layers)
        if numInputs != 2:
            raise ValueError("Euclid_dist_Layer needs 2 layers as inputs, however got {}".format(numInputs))
        if self.input_shapes[0] != self.input_shapes[1]:
            raise ValueError("Euclid_dist_Layer needs the inputs have the same shape, "
                                "however got {}".format(self.input_shapes))
        self.axis = axis
        self.keepdims = keepdims
        self.L2_square = L2_square

    def get_output_shape_for(self, input_shapes):
        input_shape_tuple = input_shapes[0]
        input_shape_list = list(input_shape_tuple)
        if self.keepdims:
            input_shape_list[self.axis] = 1
        else:
            # input_shape_list.remove(input_shape_list[self.axis]) # wrong, because multi-elements could share the same value.
            del input_shape_list[self.axis]
        return tuple(input_shape_list)

    def get_output_for(self, inputs, **kwargs):
        diff = inputs[0] - inputs[1]
        diff_sq_sm = (diff**2).sum(axis=self.axis, keepdims=self.keepdims)
        diff_Euclid = diff_sq_sm ** .5
        return diff_sq_sm if self.L2_square else diff_Euclid 

