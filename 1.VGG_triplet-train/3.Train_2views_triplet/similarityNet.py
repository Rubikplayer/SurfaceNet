"""
define network structures    
"""
import lasagne
import theano
#if lasagne.utils.theano.config.device == 'cpu':
if lasagne.utils.theano.sandbox.cuda.dnn_available(): # when cuDNN available
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer 
else:
    from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, DenseLayer, SliceLayer, ReshapeLayer, ConcatLayer, batch_norm,FlattenLayer
from lasagne.nonlinearities import tanh, sigmoid,rectify
from layers import Get_center_2x2, Norm_L2_Layer, Euclid_dist_Layer
import theano.tensor as T
import params_similNet as param

def input_var_TO_featureEmbed_layer(input_var, input_shape_h = param.__hw_size, input_shape_w = param.__hw_size):
    net = {}
    net['input'] = InputLayer((None, 3, input_shape_h, input_shape_w), \
	      input_var=input_var - param.__MEAN_IMAGE_BGR[None,:,None,None]) # only store uint8 in memory
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_3'], 2) # keep the output layer in lower dim
    
    net['concat'] = ConcatLayer([FlattenLayer(net['pool5'], 2), 
                                Get_center_2x2(net['pool1']),
                                Get_center_2x2(net['pool2']),
                                Get_center_2x2(net['pool3']),
                                Get_center_2x2(net['pool4'])
                                ], axis=1)
    net['flat1'] = FlattenLayer(net['concat'], 2)
    ##net['L2_norm'] = Norm_L2_Layer(net['flat1'])    
    ##net['featureEmbed'] = DenseLayer(net['L2_norm'], num_units=param.__featureDim, nonlinearity=None)
    # according to the faceNet paper, the embedding vectors are unit vectors!  linear+norm+triplet                          
    # https://github.com/freesouls/caffe/tree/master/facenet  
    # https://github.com/cmusatyalab/openface/blob/master/models/openface/vgg-face.def.lua
    # https://github.com/cmusatyalab/openface/blob/e0306890422d3826df448de10fa13e5f96473374/training/opts.lua
    net['dropout1'] = lasagne.layers.DropoutLayer(net['flat1'])
    net['fc1'] = DenseLayer(net['dropout1'], num_units=600, nonlinearity=rectify)
    net['dropout2'] = lasagne.layers.DropoutLayer(net['fc1'])
    # net['L2_norm'] = Norm_L2_Layer(net['fc1'])    
    net['linear1'] = DenseLayer(net['dropout2'], num_units=param.__featureDim, nonlinearity=None) # this is a linear layer
    net['L2_norm'] = Norm_L2_Layer(net['linear1'])    
    net['featureEmbed'] = net['L2_norm']
    return net

def cost_triplet(diff_pos_Euclid, diff_neg_Euclid):
    # dist = triplet_alpha - (diff_neg_sq_sum - diff_pos_sq_sum) #(diff_neg_sq_sum - diff_pos_sq_sum) / (diff_neg_Euclid - diff_pos_Euclid)
    dist = 1 - (diff_neg_Euclid/(diff_pos_Euclid+param.__triplet_alpha))
    dist_thresh = dist*(dist>0)
    return dist_thresh.sum()

def similarity_acc_cost(predict_var, similarity_cost_ON = False):
    eps = 1e-07
    N_pos_sample = predict_var.shape[0] / 2
    target_var = T.concatenate([T.zeros((N_pos_sample,1)),T.ones((N_pos_sample,1))], axis=0)

    acc = lasagne.objectives.binary_accuracy(predict_var, target_var)
    # predict_var = T.set_subtensor(predict_var[(predict_var == 0.0).nonzero()], eps)
    # predict_var = T.set_subtensor(predict_var[(predict_var == 1.0).nonzero()], 1.0 - eps)
    cost = lasagne.objectives.binary_crossentropy(predict_var, target_var).sum() if similarity_cost_ON else None
    return acc, cost   

def featureEmbed_layer_TO_similarity_layer(featureEmbed_layer, tripletInput=True):
    net = {}
    if tripletInput:
        net['reshape'] = ReshapeLayer(featureEmbed_layer, (-1,3,[1]))
        net['triplet_anchor'] = SliceLayer(net['reshape'], indices=0, axis=1) # in order to keep the dim, use slice(0,1) == array[0:1,...]
        net['triplet_pos'] = SliceLayer(net['reshape'], indices=1, axis=1)
        net['triplet_neg'] = SliceLayer(net['reshape'], indices=2, axis=1)
        net['euclid_pos'] = Euclid_dist_Layer([net['triplet_anchor'], net['triplet_pos']], axis=1, keepdims=True, L2_square = param.__squared_L2_dist)
        net['euclid_neg'] = Euclid_dist_Layer([net['triplet_anchor'], net['triplet_neg']], axis=1, keepdims=True, L2_square = param.__squared_L2_dist)
        net['euclid_dist'] = ConcatLayer([net['euclid_pos'], net['euclid_neg']],axis=0)
    else:
        net['reshape'] = ReshapeLayer(featureEmbed_layer, (-1,2,[1]))
        net['pair_1'] = SliceLayer(net['reshape'], indices=0, axis=1)
        net['pair_2'] = SliceLayer(net['reshape'], indices=1, axis=1)
        net['euclid_dist'] = Euclid_dist_Layer([net['pair_1'], net['pair_2']], axis=1, keepdims=True, L2_square = param.__squared_L2_dist)
    # input-->output (shape 1-->1), logistic regression
    net['similarity'] = DenseLayer(net['euclid_dist'], num_units=1, nonlinearity=sigmoid)
    return net
 
def similarityNet(input_var, tripletInput=True):
    """
    This is newer version of 'similarityNet_old'
    how to check the correctness. (how to check the equality of two lasagne nets)
    Can use lasagne.layers.count_params/.get_all_layers/params
    """
    net = {}
    net_featureEmbed = input_var_TO_featureEmbed_layer(input_var)
    net.update(net_featureEmbed) 

    net_similarity = featureEmbed_layer_TO_similarity_layer(net['featureEmbed'], tripletInput = tripletInput)
    net.update(net_similarity)   # dict.update the same 'key' will be 'updated'/replaced
    return net

def similarityNet_old(input_var, tripletInput=True):
    net = input_var_TO_featureEmbed_layer(input_var)
    if tripletInput:
        net['reshape'] = ReshapeLayer(net['featureEmbed'], (-1,3,[1]))
        net['triplet_anchor'] = SliceLayer(net['reshape'], indices=0, axis=1) # in order to keep the dim, use slice(0,1) == array[0:1,...]
        net['triplet_pos'] = SliceLayer(net['reshape'], indices=1, axis=1)
        net['triplet_neg'] = SliceLayer(net['reshape'], indices=2, axis=1)
        net['euclid_pos'] = Euclid_dist_Layer([net['triplet_anchor'], net['triplet_pos']], axis=1, keepdims=True)
        net['euclid_neg'] = Euclid_dist_Layer([net['triplet_anchor'], net['triplet_neg']], axis=1, keepdims=True)
        net['euclid_dist'] = ConcatLayer([net['euclid_pos'], net['euclid_neg']],axis=0)
    else:
        net['reshape'] = ReshapeLayer(net['featureEmbed'], (-1,2,[1]))
        net['pair_1'] = SliceLayer(net['reshape'], indices=0, axis=1)
        net['pair_2'] = SliceLayer(net['reshape'], indices=1, axis=1)
        net['euclid_dist'] = Euclid_dist_Layer([net['pair_1'], net['pair_2']], axis=1, keepdims=True)
    
    net['similarity'] = batch_norm(DenseLayer(net['euclid_dist'], num_units=1, nonlinearity=sigmoid))
    return net


#----------------------------------------------------------------------        
def def_updates(net, cost, layer_range_tuple_2_update, default_lr, update_algorithm='nesterov_momentum'):
    """
    learning rate for finetuning

    Parameters
    ----------
    net: dict of layers
    cost: cost function
    layer_range_tuple_2_update: ('layerName1','layerName2'), or list of tuple [(l1,l2),(l3,l4)]
            only the params within the range ('layerName1','layerName2'] will be updated
            DON'T update the 'layerName1's params
    default_lr: default lr
    update_algorithm: 'sgd' / 'nesterov_momentum'
    
    Returns
    -------
    updates: for train_fn

    Notes
    -------
    If multiple range of layers will be updated.
    Just updates_old.update(updates_new), because it is OrderedDict.
    """

    params_trainable_all = []
    if isinstance(layer_range_tuple_2_update[0], tuple):
        layer_range_tuple_2_update_iter = layer_range_tuple_2_update
    else:
        layer_range_tuple_2_update_iter = [layer_range_tuple_2_update]
    for layer_range_tuple_2_update in layer_range_tuple_2_update_iter:
        if len(layer_range_tuple_2_update) != 2:
            raise ValueError("2 element tuple is desired for layer_range_tuple_2_update, rather than {}".format(len(layer_range_tuple_2_update)))
        # params_Layer0 = [w,b], where w/b are theano tensor variable (have its own ID)
        # params_Layer1 = [w,b,w1,b1]
        # params_trainable = [w1,b1]
        params_untrainable = lasagne.layers.get_all_params(net[layer_range_tuple_2_update[0]], trainable=True)
        params_trainable = [p for p in lasagne.layers.get_all_params(net[layer_range_tuple_2_update[1]], trainable=True)\
                if not p in params_untrainable]

        print("\nonly update the weights in the range ({},{}]".format(layer_range_tuple_2_update[0], layer_range_tuple_2_update[1]))
        print("the weights to be updated: {}".format(params_trainable))

        params_trainable_all += params_trainable
            
    if update_algorithm in 'nesterov_momentum':
        layer_updates = lasagne.updates.nesterov_momentum(cost, params_trainable_all, learning_rate=default_lr, momentum=0.9)
    elif update_algorithm in 'sgd; stochastic gradient descent':
        layer_updates = lasagne.updates.sgd(cost, params_trainable_all, learning_rate=default_lr)    
    else:
        raise ValueError("the update_algorithm {} is not found".format(update_algorithm))

    return layer_updates



def def_net_fn_train_val(return_train_fn=True, return_val_fn=True):
    train_fn = None
    val_fn = None
    input_var = T.tensor4('inputs')
    net_train_val = similarityNet(input_var,tripletInput=True)

    if return_val_fn:
        predict_var_val = lasagne.layers.get_output(net_train_val['similarity'], deterministic=True)
        similarity_acc_val, _ = similarity_acc_cost(predict_var_val, similarity_cost_ON=False)
        val_fn = theano.function([input_var], [similarity_acc_val]) #similarity_acc_val

    if return_train_fn:
        l1, l2, l3, tri_anchor, tri_pos, tri_neg, diff_pos_Euclid, diff_neg_Euclid, predict_var = \
                lasagne.layers.get_output(\
                [net_train_val['flat1'], net_train_val['L2_norm'], net_train_val['featureEmbed'], \
                net_train_val['triplet_anchor'], net_train_val['triplet_pos'], net_train_val['triplet_neg'], \
                net_train_val['euclid_pos'],\
                net_train_val['euclid_neg'],\
                net_train_val['similarity']], deterministic=False)
                                                                      
        similarity_acc, similarity_cost = similarity_acc_cost(predict_var, similarity_cost_ON=True)
        tripletCost = cost_triplet(diff_pos_Euclid, diff_neg_Euclid)

        ############## cost with regularization ##############
        weight_l2_penalty = lasagne.regularization.regularize_network_params(net_train_val['similarity'], lasagne.regularization.l2) * param.__weight_decay
        cost = tripletCost + weight_l2_penalty + similarity_cost

        updates = def_updates(net=net_train_val, cost = cost, \
                layer_range_tuple_2_update=[('pool5','similarity')], default_lr=param.__DEFAULT_LR)
        train_fn = theano.function([input_var], [cost, similarity_acc, l1, l2, l3, tri_anchor, tri_pos, tri_neg, diff_pos_Euclid, diff_neg_Euclid, predict_var], updates=updates)    
        
    return net_train_val, train_fn, val_fn


def def_net_fn_test():

    input_var = T.tensor4('inputs')
    net_test = similarityNet(input_var,tripletInput=False)

    similarity = lasagne.layers.get_output(net_test['similarity'], deterministic=True)
        
    fn_test = theano.function([input_var], [similarity])  
    return net_test, fn_test


def def_patchPair_TO_feature_simil_net_fn():
    """
    Used in the fusionNet (similarityNet+volumeNet) 
    to get the embedding+similarity output with input of patch pairs
    The return the layer is used to load the trained model of the similNet
    """
    input_var = T.tensor4('inputs')
    net_fuse = similarityNet(input_var, tripletInput=False)

    feature_var, similarity = lasagne.layers.get_output([net_fuse['featureEmbed'], \
            net_fuse['similarity']], deterministic=True)

    fn_fuse = theano.function([input_var], [feature_var, similarity])  
    return net_fuse['similarity'], fn_fuse

def def_patch_TO_feature_TO_similarity_net_fn():
    """
    Used in the fusionNet
    the case where the get_patch_feature and the calc_similarity_from_features_of_patchPair are seperated.
    How to reload the trained model to these two seperated nets?
    Could use the set_all_param_values(layer_list,...)
    """
    patch_var = T.tensor4('patch')
    net_featureEmbed = input_var_TO_featureEmbed_layer(patch_var)
    patch_feature_var = lasagne.layers.get_output(net_featureEmbed['featureEmbed'], deterministic=True)
    patch2feature_fn = theano.function([patch_var], patch_feature_var)  

    featurePair_var = T.matrix('featurePair') 
    net_featurePair2simil = featureEmbed_layer_TO_similarity_layer(InputLayer((None,param.__featureDim), input_var=featurePair_var), tripletInput=False)
    featurePair_similarity_var = lasagne.layers.get_output(net_featurePair2simil['similarity'], deterministic=True)
    featurePair2simil_fn = theano.function([featurePair_var], featurePair_similarity_var)  
    
    return net_featureEmbed['featureEmbed'],net_featurePair2simil['similarity'],patch2feature_fn,featurePair2simil_fn


