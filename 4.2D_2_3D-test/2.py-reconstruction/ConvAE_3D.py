import theano.tensor as T
import theano
import sys
import os
sys.path.insert(0,'./Lasagne') # local checkout of Lasagne
import lasagne
from lasagne.layers.dnn import Conv3DDNNLayer, Pool3DDNNLayer
from lasagne.layers import ElemwiseSumLayer, ReshapeLayer, SliceLayer, ConcatLayer, batch_norm, DenseLayer
from lasagne.regularization import regularize_layer_params, l2
from theano import pp
from theano import function
import theano.tensor.nnet
import gzip
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from train_val_data import *
import random
import pickle

random.seed(201605)
np.random.seed(201605)
lr = theano.shared(np.array(5, dtype=theano.config.floatX))
lr_decay = np.array(0.1, dtype=theano.config.floatX)
lr_decay_N_epoch = 100
max_grad = 100./lr

prediction_scale = 1.3

val_visualize_ON = True
use_pretrained_model = True
model_folder = '/home/mengqi/dataset/MVS/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D-rayTrace/'
pretrained_model_file = "/home/mengqi/dataset/MVS/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/modelfiles/6.e-2D_2_3D-28-0.808_0.957.model"
#pretrained_model_file = "/home/mengqi/dataset/MVS/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D_normal/modelfiles/20160717.5.e-2D_2_3D-2-0.81_0.969.model"
if not train_ON: ## only visualize in the val process
    ##pretrained_model_file = "/home/mengqi/dataset/MVS/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D_normal/2D_2_3D-10-0.811_0.969.model"
    pretrained_model_file = "/home/mengqi/dataset/MVS/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/modelfiles/6.e-2D_2_3D-28-0.808_0.957.model"
 
if not train_ON:
    train_set = []  
if not val_ON:
    val_set = []

def load_entire_model(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    # filename = os.path.join('./', '%s.%s' % (filename, PARAM_EXTENSION))
    with open(filename) as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)

def save_entire_model(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    # filename = os.path.join('./', filename)
    # filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(model_folder+filename, 'wb') as f:
        pickle.dump(data, f)
        print("save model to: {}{}".format(model_folder, filename))

def add_noise(X):
    shape = X.shape
    X_flat = X.flatten()
    N_indx = np.random.randint(0, X.size, X.size/2) 
    N = np.random.normal(0,.5,X.size/2)
    X_flat[N_indx] = N
    
    Hole_indx = np.random.randint(0, X.size, X.size/5) 
    X_flat[Hole_indx] = 0
    return X_flat.reshape(shape)

def weighted_mult_binary_crossentropy(prediction, target, w_for_1):
    return -(w_for_1 * target * T.log(prediction) + (1.0-w_for_1)*(1.0 - target) * T.log(1.0 - prediction))

def filtered_L2_dist_cost(prediction, target, filter_01):
    return T.mul(filter_01, (prediction - target)**2 )

def filtered_innerProduct_cost(prediction_unit, target_unit, filter_01, axis=1):
    return T.mul(filter_01, (prediction_unit * target_unit).sum(axis=axis) )

class Upscale3DLayer(lasagne.layers.Layer):
    """
    based on the Upscale2DLayer
    """
    def __init__(self, incoming, scale_factor, **kwargs):
        super(Upscale3DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = (scale_factor, scale_factor, scale_factor)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor[1]
        if output_shape[4] is not None:
            output_shape[4] *= self.scale_factor[2]        
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b, c = self.scale_factor
        upscaled = input
        if c > 1:
            upscaled = T.extra_ops.repeat(upscaled, c, axis=4)        
        if b > 1:
            upscaled = T.extra_ops.repeat(upscaled, b, axis=3)
        if a > 1:
            upscaled = T.extra_ops.repeat(upscaled, a, axis=2)
        return upscaled
    
class Unpool3DLayer(lasagne.layers.Layer):
    """
    based on the Upscale2DLayer
    """
    def __init__(self, incoming, scale_factor, **kwargs):
        super(Unpool3DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = (scale_factor, scale_factor, scale_factor)
        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor[1]
        if output_shape[4] is not None:
            output_shape[4] *= self.scale_factor[2]        
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a = self.scale_factor[0]
        s0,s1,s2,_,_ = input.shape ##self.output_shape
        upscaled = T.zeros(shape=(s0,s1,s2*a,s2*a,s2*a), dtype=theano.config.floatX) # assume: a=b=c; s2=s3=s4
        ##upscaled = input
        ##upscaled = T.extra_ops.repeat(upscaled, a, axis=2)
        ##upscaled = T.extra_ops.repeat(upscaled, a, axis=3)
        ##upscaled = T.extra_ops.repeat(upscaled, a, axis=4)
        ##T.set_subtensor(upscaled,T.zeros(upscaled.shape))
        indices = [x * a + a/2 for x in T.mgrid[0:s2,0:s2,0:s2]]
        return T.set_subtensor(upscaled[:,:,indices[0],indices[1],indices[2]], input) ## T.set_subtensor has return value!!!
        
    
class ChannelPool_weightedAverage(lasagne.layers.Layer):
    def __init__(self, incoming, average_weight, **kwargs):
        super(ChannelPool_weightedAverage, self).__init__(incoming, **kwargs)
        average_weight_sum = T.shape_padright(T.sum(average_weight, axis=1), n_ones=1) + .0001
        self.channel_weight = average_weight / average_weight_sum
        
    def get_output_for(self, input, **kwargs):
        ## add 3 broadcastable dims, change 3 --> incoming.ndim - average_weight.ndim, so that it can calculate the 
        ## weighted average through multiple channels, if incoming is {N, Nviews, channel, D, D, D} / {N, Nviews, D, D, D}
        self.channel_weight = T.shape_padright(self.channel_weight,n_ones=input.ndim - self.channel_weight.ndim) 
        
        weighted_input = T.mul(input,self.channel_weight)
        op = weighted_input.sum(axis=1) # here is sum rather than mean !!! because the weight are normalized already!!!
        return op

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],) + input_shape[2:]
    
    
class ChannelPool_argmaxWeight(lasagne.layers.Layer):
    def __init__(self, incoming, average_weight, **kwargs):
        super(ChannelPool_argmaxWeight, self).__init__(incoming, **kwargs)
        self.channel_weight = average_weight
    def get_output_for(self, input, **kwargs):
        w = self.channel_weight
        op = T.shape_padaxis(input[T.arange(w.shape[0]),T.argmax(w,axis=1)], axis=1) 
        return op
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1,) + input_shape[2:]
 
     
class ChannelPool(lasagne.layers.Layer):
    def __init__(self, incoming, mode='max', **kwargs):
        super(ChannelPool, self).__init__(incoming, **kwargs)
        self.mode = mode    
    def get_output_for(self, input, **kwargs):
        if self.mode is 'max':
            return input.max(axis=1)
        elif self.mode is 'mean':
            return input.mean(axis=1)
            
        ##cube_sum = input.sum(axis=-1).sum(axis=-1).sum(axis=-1)
        ##return T.shape_padaxis(input[T.arange(cube_sum.shape[0]),T.argmax(cube_sum,axis=1)], axis=1)
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],) + input_shape[2:]
    
    
def pool_along_mask(a_mask_value, initial, mask, value):
    """ 
    cannot be defined in the class RayTrace_PoolLayer, otherwise, the # of inputs will not correct any more! 
    in each iteration:
    process all the 3D values having the same maskID
    a_mask_value: scalar in arange(NO_of_maskIDs)
    mask/value/return: {1,D,D,D}
    
    after scan/loop:
    scan's result: {NO_of_maskIDs,1,D,D,D}
    """
    # impliment: a_location = T.argmax(value[mask==a_mask_value]).
    # BUT, the value[selection] and the == don't work as expected
    # replaced by T.where(?,a,b), and T.eq(a,b)
    ##"Cannot cast True or False as a tensor variable. Please use 1 or "
                ##"0. This error might be caused by using the == operator on "
                ##"Variables. v == w does not do what you think it does, "
                ##"use theano.tensor.eq(v, w) instead."    
    selection = T.eq(mask, a_mask_value)
    masked_value = T.where(selection,value,np.asarray(0,dtype=value.dtype)) # Attention: the min value of vriable 'value' should be larger than 0.
    a_maxValue = T.max(masked_value)
    ##There is 2 behavior of numpy.where(condition, [x ,y]). Theano always support you provide 3 parameter to where(). \
    ##As said in NumPy doc[1], numpy.where(cond) is equivalent to nonzero(). To get all the nonzero indices returned can use the nonzero() function~
    #a_maxLocation = T.and_(selection, T.eq(masked_value, a_maxValue)).nonzero()
    #T.set_subtensor(initial[a_maxLocation],a_maxValue)
    selection_argmax = T.eq(masked_value, a_maxValue)
    return T.set_subtensor(initial[T.and_(selection, selection_argmax).nonzero()],a_maxValue) #T.where(T.and_(selection, selection_argmax), a_maxValue, initial)

def pool_each_sample(sample_maskID, sample_input):
    """
    in each iteration:
    process each 2D value according to its 2D mask
    sample_N_indx: scalar in arange(N)
    sample_maskID: {2,D,D,D}
    sample_input: {1,D,D,D}
    return: {1,D,D,D}
    
    after scan/loop:
    scan's result: {N,1,D,D,D}
    """
    tensor_IDmask_0 = sample_maskID[0:1] ## {2,D,D,D} --> {1,D,D,D}
    tensor_IDmask_1 = sample_maskID[1:2] 
    
    max_mask_value_0 = T.max(tensor_IDmask_0) 
    max_mask_value_1 = T.max(tensor_IDmask_1) 
    
    initial = T.zeros_like(sample_input, dtype=sample_input.dtype)    
    
    result_0, updates_0 = theano.scan(fn=pool_along_mask,\
                            sequences=T.arange(max_mask_value_0+1, dtype=tensor_IDmask_0.dtype),\
                            outputs_info=initial,\
                            non_sequences=[tensor_IDmask_0, sample_input]) ## {N,1,D,D,D} --> {1,D,D,D}
    result_1, updates_1 = theano.scan(fn=pool_along_mask,\
                            sequences=T.arange(max_mask_value_1+1, dtype=tensor_IDmask_1.dtype),\
                            outputs_info=result_0[-1],\
                            non_sequences=[tensor_IDmask_1, sample_input])
    
    #### use T.shape_padaxis(a, axis=1) OR a.dimshuffle([0,'x',1,2,3] to add dimension
    ##concat_result_0 = T.sum(result_0, axis=0)##.dimshuffle([0,'x',1,2,3]) ## list of {1,1,D,D,D} ==> {1,1,D,D,D}
    ##concat_result_1 = T.sum(result_1, axis=0)##.dimshuffle([0,'x',1,2,3])
    
    ##return T.where(concat_result_0>0, concat_result_0, concat_result_1)    
    return result_1[-1]
    
class RayTrace_PoolLayer(lasagne.layers.Layer):
    def __init__(self, incoming, tensor_IDmasks, **kwargs):
        super(RayTrace_PoolLayer, self).__init__(incoming, **kwargs)
        self.tensor_IDmasks = tensor_IDmasks
        
    def get_output_for(self, input, **kwargs):
        ## IDmasks: {N, Nviews, D, D, D}
        ## input: {N, 1, D,D,D}
        result, updates = theano.scan(fn=pool_each_sample,\
                                outputs_info=None,\
                                sequences=[self.tensor_IDmasks, input],\
                                non_sequences=None)   
        return result  ##, updates_0+updates_1

    def get_output_shape_for(self, input_shape):
        return input_shape
    
    
    
def W_5D(size):
    size = float(size)
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size, :size]
    W = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor) * \
           (1 - abs(og[2] - center) / factor)
    return W[None,None].astype(np.float32)


#----------------------------------------------------------------------
def preprocess_augmentation(list_sub, X_sub, augment_ON = True):
    X_sub /= 255.
    ##mean_rgb = np.array([122.7, 116.7, 104.0, 122.7, 116.7, 104.0])[None,:,None,None,None]
    X_sub -= .5 ##mean_rgb  ##.5
    if augment_ON:
        list_sub, X_sub = data_augment_rand_rotate(list_sub, X_sub) # randly rotate multiple times
        list_sub, X_sub = data_augment_rand_rotate(list_sub, X_sub)
        list_sub, X_sub = data_augment_rand_rotate(list_sub, X_sub)
        ##gt_sub, X_sub = data_augment_scipy_rand_rotate(gt_sub, X_sub) ## take a lot of time
        crop_result = data_augment_rand_crop(list_sub+[X_sub]) # smaller size cube    
        list_sub, X_sub = crop_result[:-1], crop_result[-1]
    return list_sub, X_sub


#----------------------------------------------------------------------
def preprocess_augmentation_withNormal(gt_sub, X_sub, gt_normal_sub, only_crop_ON = False):
    """
    Compared with the preprocess_augmentation, we add the support for the normal vector as input.
    """
    X_sub /= 255.
    ##mean_rgb = np.array([122.7, 116.7, 104.0, 122.7, 116.7, 104.0])[None,:,None,None,None]
    X_sub -= .5 ##mean_rgb  ##.5
    if not only_crop_ON:
        gt_sub, X_sub, gt_normal_sub = data_augment_rand_rotate_withNormal(gt_sub, X_sub, gt_normal_sub) # randly rotate multiple times
        gt_sub, X_sub, gt_normal_sub = data_augment_rand_rotate_withNormal(gt_sub, X_sub, gt_normal_sub)
        gt_sub, X_sub, gt_normal_sub = data_augment_rand_rotate_withNormal(gt_sub, X_sub, gt_normal_sub)
        ##gt_sub, X_sub = data_augment_scipy_rand_rotate(gt_sub, X_sub) ## take a lot of time
    gt_sub, X_sub, gt_normal_sub = data_augment_rand_crop([gt_sub, X_sub, gt_normal_sub]) # smaller size cube    
    return gt_sub, X_sub, gt_normal_sub

#----------------------------------------------------------------------
def Bilinear_3DInterpolation(incoming, upscale_factor,  
               untie_biases=False, nonlinearity=None,pad='same' ):
    """ 3Dunpool + 3DDeconv with fixed filters 
    In order to support multi-channel bilinear interpolation without extra effort, we can simply reshape it into 1-channel feature maps
    before do the interpolation followed with another reshape Layer.
    """
    unpooledLayer = Unpool3DLayer(incoming, upscale_factor)
    k_size = upscale_factor/2 * 2 + 1

    unpooledLayer_1channel = ReshapeLayer(unpooledLayer, shape=(-1, 1)+unpooledLayer.output_shape[-3:])    
    deconvedLayer = Conv3DDNNLayer(unpooledLayer_1channel,1,(k_size,k_size,k_size),nonlinearity=nonlinearity,\
                                   untie_biases=untie_biases,pad=pad,b=None,W=W_5D(k_size))
    deconvedLayer.params[deconvedLayer.W].remove('trainable')     
    
    return ReshapeLayer(deconvedLayer, shape=(-1,)+unpooledLayer.output_shape[1:])

#----------------------------------------------------------------------
def Dilated_Conv3DDNNLayer(incoming, num_filters, filter_size, dilation_size,  
               untie_biases=False, nonlinearity=None,pad='same' ):
    """ when dilation_size=1, it's the normal conv weight kernal"""
    pad_size = dilation_size-1
    num_input_channels = incoming.output_shape[1]
    dilated_filter_size = tuple([s+(s-1)*pad_size for s in filter_size])
    # W initialization
    W_init_np = np.random.normal(0,.001,(num_filters,num_input_channels)+dilated_filter_size).astype(np.float32)
    cj,ci = W_init_np.shape[0:2]
    if cj==ci:
        center_indx = dilated_filter_size[-1]/2
        for a in xrange(ci):
            W_init_np[a,a,center_indx,center_indx,center_indx] = 1
    
    W_mask = np.zeros_like(W_init_np).astype(np.uint8)

    W_mask[:,:,0::dilation_size,0::dilation_size] = 1
            
    W = theano.shared(W_init_np)

    dilated_conv_output = Conv3DDNNLayer(incoming, num_filters, dilated_filter_size, 
               untie_biases=untie_biases, W=W*W_mask, 
               nonlinearity=nonlinearity, pad=pad)
    return dilated_conv_output

#----------------------------------------------------------------------
class Norm_L2_Layer(lasagne.layers.Layer):
    def __init__(self, incoming, axis_norm, **kwargs):
        super(Norm_L2_Layer, self).__init__(incoming, **kwargs)
        self.axis_norm = axis_norm    
    def get_output_for(self, input, **kwargs):
        input_sqr = input**2
        input_L2 = (input_sqr.sum(axis=self.axis_norm))**.5 + .001
        input_unit = input/T.shape_padaxis(input_L2, axis=self.axis_norm)
        return input_unit
    

def main():
    
    chunk_len = 6
    chunk_len_val = 16
    num_epochs = 1000
    
    N_viewPairs = 6 # NO of view pair combinations
    
    if (not train_ON) and val_ON:
        chunk_len_val = 16
        
    if (not train_ON) and (not val_ON) and reconstr_ON:
        
        N_viewPairs = 8 #56 # NO of view pair combinations 
        chunk_len_test = 5 ##5 / 80
        
        dict_viewIndx_2_dimIndx = {_viewIndx:_dimIndx for _dimIndx, _viewIndx in enumerate(view_set)}
        dict_dimIndx_2_viewIndx = {_dimIndx:_viewIndx for _dimIndx, _viewIndx in enumerate(view_set)}
        
        
    #X_train_noisy, X_train_gt, X_val_noisy, X_val_gt = load_3Ddata(LowResolution=False)
    cameraPOs = load_all_cameraPO_files_f64(view_list = view_set)
    models_img = load_modellist_meanIMG(train_set+val_set+test_set)
    if train_ON or val_ON:
        train_gt_param, val_gt_param, train_gt_01, val_gt_01, train_gt_Normal, val_gt_Normal, models_densityCube_param =\
            load_train_val_gt_asnp(visualization_ON=False)
        train_VGGOccupancy, train_VGGFeature, val_VGGOccupancy, val_VGGFeature= load_train_val_VGGOccupancy_VGGFeature_asnp(train_set, val_set, models_densityCube_param)
    ##mean1D = X_train_noisy.flatten().mean()
    ##mean4D = X_train_noisy.mean(axis=0)
    ##mean5D = mean4D[None,:]
    mean = 0   # mean1D/5D
    mean_visualizer = 0    # mean1D/4D
    ##X_train_noisy -= mean
    ##X_train_gt -= mean
    ##X_val_noisy -= mean
    ##X_val_gt -= mean
    
    input_hwd = D_randcrop

    tensor_IDmask = T.TensorType('uint16',(False,)*5)('IDmask')
    tensor5D = T.TensorType('float32', (False,)*5)
    input_var = tensor5D('X')
    output_var_surf = T.TensorType('float32', (False,True)+(False,)*3)('Y_surf') ## need to be broadcastable for {N,1,D,D,D}*{N,3,D,D,D}
    if train_with_normal:
        output_var_normalXYZ = tensor5D('Y_normal')
    
    vgg_weight_01 = T.fmatrix('w')

    nonlinearity_sigmoid = lasagne.nonlinearities.sigmoid
    nonlinearity_tanh = lasagne.nonlinearities.tanh
    ##conv_nonlinearity = lasagne.nonlinearities.sigmoid
    conv_nonlinearity = lasagne.nonlinearities.rectify
    nonlinear_PReLU = lasagne.nonlinearities.LeakyRectify(.1)
    #conv_nonlinearity = theano.tensor.nnet.softplus    pred_fuse = lasagne.layers.get_output(fuse_op_reshape_channelPool)

    
    input = lasagne.layers.InputLayer((None,3*2,input_hwd,input_hwd,input_hwd), input_var)
    input_chunk_len = input_var.shape[0] / N_viewPairs
    ##input = ReshapeLayer(input, shape=(input_var.shape[0]*2, 3)+(input_hwd,)*3)
    
    #---------------------------    
    conv1_1 = batch_norm(Conv3DDNNLayer(input,32,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    conv1_2 = batch_norm(Conv3DDNNLayer(conv1_1,32,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    conv1_3 = batch_norm(Conv3DDNNLayer(conv1_2,32,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    
    pool1 = Pool3DDNNLayer(conv1_3, (2,2,2), stride=2)
    side_op1 = batch_norm(Conv3DDNNLayer(conv1_3,16,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    side_op1_deconv = side_op1
    
    #---------------------------
    conv2_1 = batch_norm(Conv3DDNNLayer(pool1,80,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    conv2_2 = batch_norm(Conv3DDNNLayer(conv2_1,80,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    conv2_3 = batch_norm(Conv3DDNNLayer(conv2_2,80,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    
    pool2 = Pool3DDNNLayer(conv2_3, (2,2,2), stride=2)  
    side_op2 = batch_norm(Conv3DDNNLayer(conv2_3,16,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    side_op2_deconv = Bilinear_3DInterpolation(side_op2, upscale_factor=2, untie_biases=False, nonlinearity=None, pad='same')
                                                  
    #---------------------------
    conv3_1 = batch_norm(Conv3DDNNLayer(pool2,160,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    conv3_2 = batch_norm(Conv3DDNNLayer(conv3_1,160,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    conv3_3 = batch_norm(Conv3DDNNLayer(conv3_2,160,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same') )
    
    ##pool3 = Pool3DDNNLayer(conv3_3, (2,2,2), stride=2)  
    side_op3 = batch_norm(Conv3DDNNLayer(conv3_3,16,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    side_op3_deconv = Bilinear_3DInterpolation(side_op3, upscale_factor=4, untie_biases=False, nonlinearity=None, pad='same')
        
    #---------------------------
    conv4_1 = batch_norm(Dilated_Conv3DDNNLayer(conv3_3,300,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    conv4_2 = batch_norm(Dilated_Conv3DDNNLayer(conv4_1,300,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    conv4_3 = batch_norm(Dilated_Conv3DDNNLayer(conv4_2,300,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same') )
    side_op4 = batch_norm(Dilated_Conv3DDNNLayer(conv4_3,16,(1,1,1),dilation_size=2,nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    side_op4_deconv = Bilinear_3DInterpolation(side_op4, upscale_factor=4, untie_biases=False, nonlinearity=None, pad='same')
        
    #---------------------------
    fuse_side_outputs = ConcatLayer([side_op1_deconv,side_op2_deconv,side_op3_deconv,side_op4_deconv], axis=1)
    fusion_conv = batch_norm(Conv3DDNNLayer(fuse_side_outputs,100,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    fusion_conv = batch_norm(Conv3DDNNLayer(fusion_conv,100,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    fusion_conv_surf = batch_norm(Conv3DDNNLayer(fusion_conv,1,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    ##fusion_DilatedConv = batch_norm(Dilated_Conv3DDNNLayer(fuse_side_outputs,100,(3,3,3),dilation_size=1,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_DilatedConv = batch_norm(Dilated_Conv3DDNNLayer(fusion_DilatedConv,100,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_DilatedConv = batch_norm(Dilated_Conv3DDNNLayer(fusion_DilatedConv,100,(3,3,3),dilation_size=4,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_conv2 = batch_norm(Dilated_Conv3DDNNLayer(fusion_DilatedConv,100,(3,3,3),dilation_size=1,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_conv3 = batch_norm(Dilated_Conv3DDNNLayer(fusion_conv2,1,(3,3,3),dilation_size=1,nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same')    )
    #---------------------------
    #fusion_conv_surf = RayTrace_PoolLayer(fusion_conv_surf, tensor_IDmask) 
    
    fuse_op_reshape_surf = ReshapeLayer(fusion_conv_surf, shape=(input_chunk_len, N_viewPairs, 1, input_hwd,input_hwd,input_hwd))
    
    
    if train_with_normal:
        fusion_conv_normalXYZ = batch_norm(Conv3DDNNLayer(fusion_conv,3,(1,1,1),nonlinearity=nonlinearity_tanh,untie_biases=False,pad='same'))
        fuse_op_reshape_normalXYZ = Norm_L2_Layer(ReshapeLayer(fusion_conv_normalXYZ, shape=(input_chunk_len, N_viewPairs, 3, input_hwd,input_hwd,input_hwd)),axis_norm=2)
    
    ##average_weight = batch_norm(Conv3DDNNLayer(conv4_3,1,conv4_3.output_shape[-3:],nonlinearity=nonlinearity_sigmoid))
    ##average_weight = ReshapeLayer(average_weight, shape=(input_chunk_len, N_viewPairs)) ##, W=lasagne.init.Constant(.1)
    ##average_weight_tensor = lasagne.layers.get_output(average_weight)
    
    ##fuse_op_reshape_channelPool = ChannelPool_argmaxWeight(fuse_op_reshape, average_weight_tensor)
    if reconstr_ON:
        fuse_op_reshape_channelPool_surf = ChannelPool_weightedAverage(fuse_op_reshape_surf, vgg_weight_01)
        if train_with_normal:
            fuse_op_reshape_channelPool_normalXYZ = ChannelPool_weightedAverage(fuse_op_reshape_normalXYZ, vgg_weight_01)
    else:
        fuse_op_reshape_channelPool_surf = ChannelPool(fuse_op_reshape_surf, mode='max')
        if train_with_normal:
            fuse_op_reshape_channelPool_normalXYZ = ChannelPool(fuse_op_reshape_normalXYZ, mode='mean')
    
    output_layer_surf = fuse_op_reshape_channelPool_surf ##fuse_op_reshape_channelPool / conv1_3
    if train_with_normal:
        output_layer_normalXYZ = fuse_op_reshape_normalXYZ 
        output_layer_normalXYZ_fuse_unit = Norm_L2_Layer(fuse_op_reshape_channelPool_normalXYZ,axis_norm=1) 
    ##net_pair = ReshapeLayer(net, shape=(input_var.shape[0], net.output_shape[1]*2, input_hwd,input_hwd,input_hwd))
    ##net = batch_norm(Conv3DDNNLayer(net_pair,1,(3,3,3),nonlinearity=lasagne.nonlinearities.sigmoid,untie_biases=False,pad='same') # enable some negative values
    
    #---------------------------
    if train_with_normal:
        params = lasagne.layers.get_all_params([output_layer_surf, output_layer_normalXYZ], trainable=True) #
    else:
        params = lasagne.layers.get_all_params(output_layer_surf, trainable=True)
    print "surf output shape:",output_layer_surf.output_shape
    pred_fuse_surf = lasagne.layers.get_output(output_layer_surf)
    if train_with_normal:
        print "normalXYZ output shape:",output_layer_normalXYZ.output_shape
        pred_fuse_normalXYZ = lasagne.layers.get_output(output_layer_normalXYZ)
        pred_fuse_normalXYZ_fuse_unit = lasagne.layers.get_output(output_layer_normalXYZ_fuse_unit)

    ##prediction = T.shape_padaxis(prediction.max(axis=1), axis=1) ## pool along channels
    ##prediction = conv_nonlinearity(prediction) # perform the nonlinearity on elementwiseSum
    
    epsi = .1
    ##loss = (prediction+epsi)/(output_var+epsi) + (output_var+epsi)/(prediction+epsi)
    ##loss = lasagne.objectives.binary_crossentropy(prediction, output_var)
    loss_surf = weighted_mult_binary_crossentropy(pred_fuse_surf, output_var_surf, w_for_1 = 0.97) \
        + regularize_layer_params(output_layer_surf,l2) * 1e-4
    aggregated_loss_surf = lasagne.objectives.aggregate(loss_surf)
         
    if train_with_normal:    
        loss_normalXYZ = filtered_L2_dist_cost(pred_fuse_normalXYZ, T.shape_padaxis(output_var_normalXYZ,axis=1), filter_01=T.shape_padaxis(output_var_surf,axis=1)) \
            + regularize_layer_params(output_layer_normalXYZ,l2) * 1e-4
        aggregated_loss_normalXYZ = lasagne.objectives.aggregate(loss_normalXYZ, weights=5.)
        grads = theano.grad(aggregated_loss_surf + aggregated_loss_normalXYZ , params) #
    else:
        grads = theano.grad(aggregated_loss_surf, params) #
        
    grads = [T.clip(g, -1*max_grad, max_grad) for g in grads]
    updates = lasagne.updates.nesterov_momentum(grads, params, learning_rate=lr) ##+ updates_rayTrace    
    pred_fuse_surf_val = lasagne.layers.get_output(fuse_op_reshape_channelPool_surf, deterministic=True)
    if train_ON:
        accuracy = lasagne.objectives.binary_accuracy(pred_fuse_surf, output_var_surf)
        if train_with_normal:  
            train_fn = theano.function([input_var, output_var_surf, output_var_normalXYZ], \
                                       [aggregated_loss_surf, aggregated_loss_normalXYZ,accuracy, pred_fuse_surf, pred_fuse_normalXYZ_fuse_unit], updates=updates)
        else:
            train_fn = theano.function([input_var, tensor_IDmask, output_var_surf], \
                                        [aggregated_loss_surf,accuracy, pred_fuse_surf], updates=updates)
            
    if val_ON:
        accuracy_val = lasagne.objectives.binary_accuracy(pred_fuse_surf_val, output_var_surf)
        if train_with_normal:          
            val_fn = theano.function([input_var, output_var_surf], \
                                     [accuracy_val,pred_fuse_surf_val, pred_fuse_normalXYZ_fuse_unit])
        else:
            val_fn = theano.function([input_var, tensor_IDmask, output_var_surf], [accuracy_val,pred_fuse_surf_val])            
    if reconstr_ON:
        input_rgb_pair = ReshapeLayer(input, shape=(input_chunk_len, N_viewPairs, 3*2, input_hwd,input_hwd,input_hwd))
        input_rgb_pair_weighted = lasagne.layers.get_output(ChannelPool_weightedAverage(input_rgb_pair, vgg_weight_01)).squeeze()
        input_rgb = (input_rgb_pair_weighted[:,:3] + input_rgb_pair_weighted[:,3:])/2 ## {N,3,D,D,D}
        if train_with_normal:     
            test_fn = theano.function([input_var, vgg_weight_01], [pred_fuse_surf_val, input_rgb, pred_fuse_normalXYZ_fuse_unit])
        else:
            test_fn = theano.function([input_var, vgg_weight_01], [pred_fuse_surf_val, input_rgb])
            
#===============================================
    # load the pretrained model
    if use_pretrained_model == True:
        print ('loading model: {}'.format(pretrained_model_file))
        if train_ON or (not train_with_normal):
            load_entire_model([output_layer_surf], pretrained_model_file)      ##[output_layer_surf,output_layer_normalXYZ]
        else:
            load_entire_model([output_layer_surf, output_layer_normalXYZ], pretrained_model_file)      

#===============================================        
    print "starting training..."
    start_time_train_val = time.time()
    if train_ON:
        train_gt_param, train_inScope_indx = inScope_nearSurf_Cubes(N_inScopeViews = len(view_set)-1, occupiedCubes_param = train_gt_param, \
                                     VGGOccupancy = train_VGGOccupancy, cameraPOs=cameraPOs, models_img=models_img)   
        train_gt_01, train_VGGOccupancy, train_VGGFeature = \
            train_gt_01[train_inScope_indx], train_VGGOccupancy[train_inScope_indx], train_VGGFeature[train_inScope_indx]
        if train_with_normal:  
            train_gt_Normal = train_gt_Normal[train_inScope_indx]
         
        N_train = train_gt_01.shape[0]
    if val_ON:
        val_gt_param, val_inScope_indx = inScope_nearSurf_Cubes(N_inScopeViews = len(view_set)-1, occupiedCubes_param = val_gt_param, \
                                      VGGOccupancy = val_VGGOccupancy, cameraPOs=cameraPOs, models_img=models_img)        
        val_gt_01, val_VGGOccupancy, val_VGGFeature = \
            val_gt_01[val_inScope_indx], val_VGGOccupancy[val_inScope_indx], val_VGGFeature[val_inScope_indx]
        if train_with_normal:  
            val_gt_Normal = val_gt_Normal[val_inScope_indx]
        N_val = val_gt_01.shape[0]
    for epoch in range(1, num_epochs):
        
        if train_ON:
            if epoch%lr_decay_N_epoch == 0:
                lr.set_value(lr.get_value() * lr_decay)        
                print 'current updated lr = {}'.format(lr.get_value())
            
            print "starting VGG_triplet Net to predict the feature vectors for each view patches."
                
            acc_train_batches = []
            acc_guess_all0 = []
            _aggregated_loss_normalXYZ = 0
            
            for batch in range(1, N_train/chunk_len): ##3 or N_train/chunk_len
                selected = random.sample(range(0,N_train),chunk_len) ##almost like shuffle
                ##selected = list(set(np.random.random_integers(0,N_train-1,chunk_len*2)))[:chunk_len] ## set([2,2,3])=[2,3], 
                train_gt_sub = train_gt_01[selected][:,None,...] ## convert to 5D
                if train_with_normal: 
                    train_gt_Normal_sub = train_gt_Normal[selected] ## 5D
                ## do some simple data augmentation in each batch, save back to the data_array, which will be shuffled next time
                #train_X_sub, selected_viewPair_dimIndx_np, VGG_triplet_Net_weight_01 = gen_coloredCubes(N_views = N_views, N_viewPairs = N_viewPairs, occupiedCubes_param = train_gt_param[selected], \
                                 #cameraPOs=cameraPOs, models_img=models_img, visualization_ON = False, occupiedCubes_01 = train_gt_sub, \
                                 #feature_embed = train_VGGFeature[selected])
                train_X_sub, train_IDmask_sub = gen_coloredCubes_withoutVGGTriplet(N_views = N_views, N_viewPairs = N_viewPairs, occupiedCubes_param = train_gt_param[selected], \
                                 cameraPOs=cameraPOs, models_img=models_img, visualization_ON = False, occupiedCubes_01 = train_gt_sub, \
                                 return_pixIDmask_cubes_ON=True)
                if train_with_normal: 
                    train_gt_sub, train_X_sub, train_gt_Normal_sub = preprocess_augmentation_withNormal(train_gt_sub, train_X_sub, gt_normal_sub = train_gt_Normal_sub)
                    _aggregated_loss_surf, _aggregated_loss_normalXYZ, acc, predict_train, predict_normal_train = train_fn(train_X_sub, train_gt_sub, train_gt_Normal_sub)
                else:
                    [train_gt_sub,train_IDmask_sub], train_X_sub = preprocess_augmentation([train_gt_sub,train_IDmask_sub], train_X_sub)
                    _aggregated_loss_surf, acc, predict_train = train_fn(train_X_sub, train_IDmask_sub, train_gt_sub)
                    
                acc_train_batches.append(acc)
                acc_guess_all0.append(1-float(train_gt_sub.sum())/train_gt_sub.size)
                print("Epoch %d, batch %d: Loss1 %g, Loss2 %g, acc %g, acc_guess_all0 %g" % \
                                              (epoch, batch, _aggregated_loss_surf, _aggregated_loss_normalXYZ, np.asarray(acc_train_batches).mean(), np.asarray(acc_guess_all0).mean()))
                
            if (epoch % 2) == 0:
                if train_with_normal: 
                    save_entire_model([output_layer_surf, output_layer_normalXYZ], '2D_2_3D-{}-{:0.3}_{:0.3}.model'.format(epoch,np.asarray(acc_train_batches).mean(),np.asarray(acc_guess_all0).mean()))             
                else:    
                    save_entire_model([output_layer_surf], '2D_2_3D-{}-{:0.3}_{:0.3}.model'.format(epoch,np.asarray(acc_train_batches).mean(),np.asarray(acc_guess_all0).mean()))             
        
        if val_ON:       
            if (epoch % 1) == 0:    
                print "starting validation..."    
                print "starting VGG_triplet Net to predict the feature vectors for each view patches."
                
                acc_val_batches = []
                for batch_val in range(0, N_val/chunk_len_val):
                    selected = range(batch_val*chunk_len_val,(batch_val+1)*chunk_len_val)
                    val_gt_sub = val_gt_01[selected][:,None,...] ## convert to 5D
                    if train_with_normal: 
                        val_gt_Normal_sub = val_gt_Normal[selected] ## 5D
                    #val_X_sub, selected_viewPair_dimIndx_np, VGG_triplet_Net_weight_01 = gen_coloredCubes(N_views = N_views, N_viewPairs = N_viewPairs, occupiedCubes_param = val_gt_param[selected], \
                                                 #cameraPOs=cameraPOs, models_img=models_img, visualization_ON = False, occupiedCubes_01 = val_gt_sub, \
                                                 #feature_embed = val_VGGFeature[selected])     
                    val_X_sub, val_IDmask_sub = gen_coloredCubes_withoutVGGTriplet(N_views = N_views, N_viewPairs = N_viewPairs, occupiedCubes_param = val_gt_param[selected], \
                                                                     cameraPOs=cameraPOs, models_img=models_img, visualization_ON = False, occupiedCubes_01 = val_gt_sub,\
                                                                     return_pixIDmask_cubes_ON=True)                         
                    if train_with_normal: 
                        val_gt_sub, val_X_sub, val_gt_Normal_sub = preprocess_augmentation_withNormal(val_gt_sub, val_X_sub, val_gt_Normal_sub, only_crop_ON=False)
                        ##val_gt_sub, val_X_sub, val_gt_Normal_sub = data_augment_rand_crop([val_gt_sub, val_X_sub, val_gt_Normal_sub]) # smaller size cube    
                        
                        acc_val, predict_val, predict_normal_val = val_fn(val_X_sub, val_gt_sub)
                    else:                        
                        [val_gt_sub,val_IDmask_sub], val_X_sub = preprocess_augmentation([val_gt_sub,val_IDmask_sub], val_X_sub)
                        acc_val, predict_val = val_fn(val_X_sub, val_IDmask_sub, val_gt_sub)
                    acc_val_batches.append(acc_val)   
                    if val_visualize_ON :
                        X_1, X_2  = val_X_sub[0:N_viewPairs], val_gt_sub[0:1]
                        if train_with_normal: 
                            N_1, N_2 = predict_normal_val[0:1], val_gt_Normal_sub[0:1]
                        result = predict_val[0]
                        X_1 = (X_1+.5)*255.
                        tmp_5D = np.copy(X_1) # used for visualize the surface part of the colored cubes
                        # if want to visualize the result, just stop at the following code, and in the debug probe run the visualize_N_densities_pcl func.
                        # rimember to 'enter' before continue the program~
                        occupancy = X_2[0].squeeze()!=0
                        ##tmp_5D[:,:,X_2[0].squeeze()==0]=0 
                        if not train_ON:
                            visualize_N_densities_pcl([X_2[0]*2, result*prediction_scale, tmp_5D[0,3:], tmp_5D[1,3:], X_1[0,3:], X_1[1,3:]],\
                                                      normalCube_list = [val_gt_Normal_sub[0:1], predict_normal_val[0:1],None,None,None,None] if train_with_normal else [None]*6, \
                                                      density_list=[None,None,occupancy,occupancy,None,None])
                print("val_acc %g" %(np.asarray(acc_val_batches).mean()))
                
        print("\nit took {:.3f}s\n".format(time.time() - start_time_train_val))
        if not train_ON: ## only run one epoch
            break
        
    print "train & val done."
    
#=============================================== 
    if reconstr_ON:  
        print "starting reconstruction..."    
        start_time = time.time()
        points = [] 
        saved_params = []
        saved_prediction = []
        saved_maskID = []
        saved_selected_pairIndx = []
        saved_rgb = []
        Samples_param_np, cube_Dsize = RECONSTR_generate_3DSamples_param_np(test_set[0])
        ##Nx,Ny,Nz = Samples_param_np[5].shape ## Nx * Ny * Nz = # of 3D cubes
        #### store the ix/iy/iz th cube's points in points_3Dlist[ix][iy][iz]. 
        ##refine_cubes_points_3Dlist = [[[[] for z in xrange(Nz)] for y in xrange(Ny)] for x in xrange(Nx)] 
        ##refine_cubes_occupancy = np.zeros((Nx,Ny,Nz)).astype(np.bool)
        ##refine_cubes_NOofParams = 3   ## adaptive threshld; # of nonempty neighbors; # of connected cubes
        ##refine_params = np.zeros((Nx,Ny,Nz,refine_cubes_NOofParams))
        
        #np.random.shuffle(Samples_param_np)
        print "starting VGG_triplet Net to predict the feature vectors for each view patches."
        
        pt_batch_N = 10000
        _iterations, _remainder = Samples_param_np.shape[0] / pt_batch_N, Samples_param_np.shape[0] % pt_batch_N
        for _i in range(0, _iterations+1): ## divide into multiple parts to reconstruct, because of the limited memory~
            if (_i ==  _iterations) and (_remainder != 0):
                Samples_select = range(_i*pt_batch_N, Samples_param_np.shape[0])   
            elif (_i ==  _iterations) and (_remainder == 0):
                continue
            else:            
                Samples_select = range(_i*pt_batch_N, (_i+1)*pt_batch_N)  
            _param = Samples_param_np[Samples_select]
        
            diff_X_Euclid, desicion_onSurf = get_VGG_triplet_Net_featureVec(modelIndx=test_set[0],view_set=view_set,\
                                                        xyz_np = _param[:,:3] + cube_Dsize/2. ) 
                                                        #xyz_nplist=[i + D_center*_param[:,3]/2 for i in [_param[:,0],_param[:,1],_param[:,2]]])
            test_param, inScope_indices = inScope_nearSurf_Cubes(N_inScopeViews = 3, occupiedCubes_param = _param, \
                                          VGGOccupancy=desicion_onSurf, cameraPOs=cameraPOs, models_img=models_img)           
            
            diff_X_Euclid, desicion_onSurf = diff_X_Euclid[inScope_indices], desicion_onSurf[inScope_indices]
            N_test = test_param.shape[0]
            
            print '\nThere are {} inScope_nearSurf_Cubes will be processed. {} / {}'.format(N_test, _i, _iterations)
            if N_test == 0:
                continue
            
            for batch_test in range(0, N_test/chunk_len_test + 1): 
                if (batch_test ==  N_test/chunk_len_test) and ((N_test%chunk_len_test) != 0):
                    selected = range(batch_test*chunk_len_test, N_test)   
                elif (batch_test ==  N_test/chunk_len_test) and ((N_test%chunk_len_test) == 0):
                    continue
                else:
                    selected = range(batch_test*chunk_len_test, (batch_test+1)*chunk_len_test)        
                test_X_sub, selected_viewPair_dimIndx_np, VGG_triplet_Net_weight_01 = gen_coloredCubes(view_set = view_set, N_viewPairs = N_viewPairs, occupiedCubes_param = test_param[selected], \
                                cameraPOs=cameraPOs, models_img=models_img, visualization_ON = False, occupiedCubes_01 = None, diff_X_Euclid = diff_X_Euclid[selected],\
                                return_pixIDmask_cubes_ON=False)                    
                _, test_X_sub = preprocess_augmentation(None, test_X_sub, augment_ON=False)
                # TODO: this prediction process has lot redundant computation: if there are only few view pairs are valid, many view pairs will be calculated and latter filtered by 0/1_weight.
                # a good solution is that: if there are few view pairs, only let these valid views go through the network.
                predict_test, rgb_test = test_fn(test_X_sub,VGG_triplet_Net_weight_01)
        
                ##points_sub = RECONSTR_select_valid_pts_in_cube(predict_test*prediction_scale, rgb_test, test_param[selected])
                points_sub = RECONSTR_select_valid_pts_in_cube(predict_test*prediction_scale, rgb_test, test_param[selected],\
                                                                desicion_onSurf[selected])
                points.append(points_sub)
                
                # points_filtered is only used to save storage. If the point is still not selected when the threshold is very loose, then we don't need to save the points in this bunch           
                if predict_test.max() < 1./16: ## not optimal, this method can only filter entire test_bunch, cannot indivisually select each sample
                    continue
                saved_prediction.append(predict_test)
                saved_rgb.append(rgb_test)
                saved_params.append(test_param[selected])
                saved_selected_pairIndx.append(map_dimIndx_2_viewIndx_np(selected_viewPair_dimIndx_np))
                #saved_maskID.append(test_IDmask_sub)
                #visualize_N_densities_pcl([predict_test[0]*1.5,predict_test[1]*1.5,predict_test[2]*1.5])
                      
            ##if ((batch_test % 100) == 1) or (batch_test == N_test/chunk_len_test):
            filename = 'model{:03}-batch-{}_{}.pcd'.format(test_set[0], _i, _iterations)
            RECONSTR_save_pcd(np.concatenate(points, axis=0), pcd_folder, filename)
            os.system( "{} {} {}".format(pcl_pcd2ply_exe, pcd_folder+filename,\
                                   (pcd_folder+filename).replace('.pcd','.ply')) )
            print("\nit took {:.3f}s\n".format(time.time() - start_time))
             
            ## directly save compressed numpy: savez_compressed
            
            save_folder = '/home/mengqi/dataset/MVS/samplesVoxelVolume/init_pcds/saved_init-{}views-{}pairs/'.\
                format(N_views, N_viewPairs)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)              
            with open(save_folder+'saved_prediction_rgb_params_model{}-{}_{}.npz'.format(test_set[0],_i, _iterations), 'wb') as f:
                np.savez_compressed(f, prediction=np.concatenate(saved_prediction,axis=0).astype(np.float16), \
                                    rgb=np.concatenate(saved_rgb,axis=0).astype(np.float16), param=np.concatenate(saved_params,axis=0),\
                                    selected_pairIndx=np.concatenate(saved_selected_pairIndx,axis=0).astype(np.uint16))
                print 'saved npz file: ', f.name
            saved_params = [] ## because of the limited memory! should release the memory after save!!!
            saved_prediction = []
            saved_rgb = []  
            saved_selected_pairIndx = []
            ##with open('/home/mengqi/dataset/MVS/samplesVoxelVolume/init_pcds/saved_prediction_rgb_params_model{}-{}_{}.msh'.format(test_set[0],_i, _iterations), 'wb') as f:
                ##marshal.dump([saved_prediction,saved_rgb,saved_params], f) # marshal doesn't support numpy
                ##print("save [saved_prediction,saved_params]")                              
        
if __name__ == "__main__":
    main()
