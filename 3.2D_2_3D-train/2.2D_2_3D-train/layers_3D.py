""" 
define some customized layers 
only loaded by *Net.py files
"""
import lasagne
from lasagne.layers import ReshapeLayer, Upscale3DLayer
from lasagne.layers.dnn import Conv3DDNNLayer
from lasagne.utils import as_tuple
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import gpu_contiguous, GpuDnnConvDesc, gpu_alloc_empty, GpuDnnConv3dGradW
import numpy as np

#========================
#   3D dilation conv
#========================
def dnn_gradweight3D(img, topgrad, imshp, kshp, 
            subsample, border_mode='valid',batchsize=None,
            filter_flip=False):
    print ('now inside dnn_gradweight3D')        

    """
    GPU convolution gradient with respect to weight using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    """
    
    if filter_flip:
        conv_mode = 'conv'
    else:
        conv_mode = 'cross'
    
    img = gpu_contiguous(img)
    topgrad = gpu_contiguous(topgrad)
    #Many tensor Ops run their arguments through this function as pre-processing.
    #It passes through TensorVariable instances,
    #and tries to wrap other objects into TensorConstant.
    kerns_shp = theano.tensor.as_tensor_variable(kshp)
    kerns_shp = [kerns_shp[0],batchsize,kerns_shp[2],kerns_shp[3],kerns_shp[4]]
    kerns_shp = theano.tensor.as_tensor_variable(kerns_shp)
    ## theano.tensor.set_subtensor(kerns_shp[1], batchsize)
    print ('kshp = {}'.format(kshp))
    print ('type = {}'.format(type(kshp)))
    print ('kerns_shp (1D shape tensor ?) = {}'.format(kerns_shp))
    ## print (' kerns_shp.ndim = {}'.format(kerns_shp.ndim))
    ## print (' kern_shape.type.dtype (int64?)= {}'.format(kerns_shp.type.dtype))
#    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
#                          conv_mode=conv_mode)(img.shape, kerns_shp)
#    desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
#                              conv_mode='cross', precision=precision)(img.shape,
#                                                                      out.shape)
#    
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(img.shape, kerns_shp)
    out = gpu_alloc_empty(*kerns_shp)
    return GpuDnnConv3dGradW()(img, topgrad, out,
                                      desc)

class DilatedConv3DLayer(lasagne.layers.DilatedConv2DLayer):
    def __init__(self, incoming, num_filters, filter_size, dilation=(1, 1, 1),
                 pad=0, untie_biases=False,
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False,
                 **kwargs):
        self.dilation = as_tuple(dilation, 3, int)
        super(lasagne.layers.DilatedConv2DLayer, self).__init__(
                incoming, num_filters, filter_size, 1, pad,
                untie_biases, W, b, nonlinearity, flip_filters, n=3, **kwargs)
        # remove self.stride:
        del self.stride
        self.batchsize = lasagne.layers.get_output(incoming).shape[0]
        # require valid convolution
        if self.pad != (0,0,0):
            raise NotImplementedError(
                    "DilatedConv2DLayer requires pad=0 / (0,0) / 'valid', but "
                    "got %r. For a padded dilated convolution, add a PadLayer."
                    % (pad,))
        # require unflipped filters
        if self.flip_filters:
            raise NotImplementedError(
                    "DilatedConv2DLayer requires flip_filters=False.")
    def convolve(self, input, **kwargs):
        # we perform a convolution backward pass wrt weights,
        # passing kernels as output gradient
        print ('calling convolve by Dilated 3D layer')
        imshp = self.input_shape
        kshp = self.output_shape
        print ('type(kshp) = {}'.format(type(kshp)))
        print ('shape of kshp = {}'.format(kshp))
        # only works with int64
        kshp_64 = np.asarray(kshp)
        # swapping
        channels = kshp[1]
        batchsize =  1
        
        kshp_64[0] = channels
        kshp_64[1] = batchsize
        kshp_64 = kshp_64.astype(np.int64)
        print ('shape of kshp_64 = {}'.format(kshp_64))#kshp = (2, 1, 15, 15, 15)
        # and swapping channels and batchsize
        imshp = (imshp[1], imshp[0]) + imshp[2:]
        #kshp = (kshp[1], kshp[0]) + kshp[2:]
        print ('should be 1D tensor, shape of kshp = {}'.format(kshp))
       
        output_size = self.output_shape[2:]
        if any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]

        conved = dnn_gradweight3D(input.transpose(1, 0, 2, 3, 4), self.W, imshp, kshp_64,
                                  self.dilation, batchsize=self.batchsize)
        return conved.transpose(1, 0, 2, 3, 4)


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
    
class ChannelPool_weightedAverage(lasagne.layers.MergeLayer):
    def __init__(self, incoming, **kwargs):
        super(ChannelPool_weightedAverage, self).__init__(incoming, **kwargs)
        
    def get_output_for(self, inputs, **kwargs):
        ## add 3 broadcastable dims, change 3 --> incoming.ndim - average_weight.ndim, so that it can calculate the 
        ## weighted average through multiple channels, if incoming is {N, Nviews, channel, D, D, D} / {N, Nviews, D, D, D}
        input = inputs[0]
        average_weight = inputs[1]
        average_weight_sum = T.shape_padright(T.sum(average_weight, axis=1), n_ones=1)
        self.channel_weight = average_weight / average_weight_sum
        self.channel_weight = T.shape_padright(self.channel_weight,n_ones=input.ndim - self.channel_weight.ndim) 
        
        weighted_input = T.mul(input,self.channel_weight)
        op = T.shape_padaxis(weighted_input.sum(axis=1), axis=1) # here is sum rather than mean !!! because the weight are normalized already!!!
        return op

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0],1,) + input_shapes[0][2:]
    
    
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
 
     
class ChannelPool_max(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return T.shape_padaxis(input.max(axis=1), axis=1)
        ##cube_sum = input.sum(axis=-1).sum(axis=-1).sum(axis=-1)
        ##return T.shape_padaxis(input[T.arange(cube_sum.shape[0]),T.argmax(cube_sum,axis=1)], axis=1)
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1,) + input_shape[2:]
    

#----------------------------------------------------------------------
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

def Bilinear_3DInterpolation(incoming, upscale_factor,  
               untie_biases=False, nonlinearity=None,pad='same' ):
    """ 3Dunpool + 3DDeconv with fixed filters 
    In order to support multi-channel bilinear interpolation without extra effort, we can simply reshape it into 1-channel feature maps
    before do the interpolation followed with another reshape Layer.
    """
    unpooledLayer = Upscale3DLayer(incoming, upscale_factor, mode='dilate') # new api from lasagne
    # unpooledLayer = Unpool3DLayer(incoming, upscale_factor) # old API
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


