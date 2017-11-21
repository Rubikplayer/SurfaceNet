import lasagne
import lasagne as L
import numpy as np
import theano
from myLayers import DilatedConv3DLayer

np.random.seed(201610)
minibatch_size = 1#2
tree_side = 5 #17
num_filters = 1
# input_data = np.random.rand(minibatch_size, 1, tree_side,tree_side,tree_side)
input_data = np.random.randint(0,9,(minibatch_size, 1, tree_side,tree_side,tree_side))
# GPU takes floatX/ 32
input_data = input_data.astype(theano.config.floatX)
net = {}

net['input'] = L.layers.InputLayer((minibatch_size,
                                        1,
                                        tree_side,tree_side,tree_side),
                                        input_var=input_data)

    
net['normal'] = lasagne.layers.dnn.Conv3DDNNLayer(net['input'], num_filters,
                                                    filter_size=(3,3,3),
                                                    W=L.init.GlorotUniform(),
                                                    b=L.init.Constant(0.),
                                                    nonlinearity=L.nonlinearities.rectify,
                                                    pad='same')

# get filter and bias
fil = lasagne.layers.get_all_params(net['normal'])
W_new = (fil[0].get_value()>0.1).astype(theano.config.floatX) # make sure it is sparse for manual computation

net['normal_new'] = lasagne.layers.dnn.Conv3DDNNLayer(net['input'], num_filters,
                                                    filter_size=(3,3,3),
                                                    W=W_new,
                                                    b=fil[1].get_value(),
                                                    nonlinearity=L.nonlinearities.rectify,
                                                    pad='same')

# feed the same filter and bias
net['dilated'] = DilatedConv3DLayer(net['input'], num_filters,
                                                    filter_size=(3,3,3),
                                                    dilation=(2,2,2),
                                                    W=W_new,
                                                    b=fil[1].get_value(),
                                                    nonlinearity=L.nonlinearities.rectify,
                                                    pad='same')

# convole input
convol_output = L.layers.get_output(net['normal_new']).eval()
dilated_output = L.layers.get_output(net['dilated']).eval()

# expect outputs are equal
# becasue dilated convolution of factor 1 = normal convolution

print('input_data: {}   w: {}   b: {}'.format(input_data, W_new, fil[1].get_value()))
print('convol_output: {}   dilated_output: {}'.format(convol_output, dilated_output))

# print (np.allclose(dilated_output, convol_output, atol=1e-07)) #only used when dilation=(1,1,1)


