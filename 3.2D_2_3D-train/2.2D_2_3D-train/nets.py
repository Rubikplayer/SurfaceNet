import lasagne
from lasagne.layers.dnn import Conv3DDNNLayer, Pool3DDNNLayer
from lasagne.layers import ElemwiseSumLayer, ReshapeLayer, SliceLayer, ConcatLayer, batch_norm, DenseLayer, NonlinearityLayer, PadLayer, ElemwiseMergeLayer
from lasagne.regularization import regularize_layer_params, l2
from layers_3D import ChannelPool_max, ChannelPool_argmaxWeight, ChannelPool_weightedAverage, Bilinear_3DInterpolation, DilatedConv3DLayer, Dilated_Conv3DDNNLayer
import theano.tensor as T
import theano
from theano.ifelse import ifelse
import numpy as np
import params_volume
from theano.tensor.shared_randomstreams import RandomStreams
T_srng = RandomStreams(seed=201704)

def volume_net_side(input_var_5D, input_var_shape,\
        N_predicts_perGroup = 6):
    """
    from the 5D input (N_cubePair, 2rgb, h, w, d) of the colored cubePairs 
    to predicts occupancy probability output (N_cubePair, 1, h, w, d)
    """
    input_var = input_var_5D
    net={}
    net["volume_input"] = lasagne.layers.InputLayer(input_var_shape, input_var)
    input_chunk_len = input_var.shape[0] / N_predicts_perGroup

    conv_nonlinearity = lasagne.nonlinearities.rectify
    nonlinearity_sigmoid = lasagne.nonlinearities.sigmoid

    #---------------------------    
    net["volume_conv1_1"] = batch_norm(Conv3DDNNLayer(net["volume_input"],32,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv1_2"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_1"],32,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv1_3"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_2"],32,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))

    net["volume_pool1"] = Pool3DDNNLayer(net["volume_conv1_3"], (2,2,2), stride=2)
    net["volume_side_op1"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_3"],16,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    net["volume_side_op1_deconv"] = net["volume_side_op1"]

    #---------------------------
    net["volume_conv2_1"] = batch_norm(Conv3DDNNLayer(net["volume_pool1"],80,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv2_2"] = batch_norm(Conv3DDNNLayer(net["volume_conv2_1"],80,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv2_3"] = batch_norm(Conv3DDNNLayer(net["volume_conv2_2"],80,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))

    net["volume_pool2"] = Pool3DDNNLayer(net["volume_conv2_3"], (2,2,2), stride=2)  
    net["volume_side_op2"] = batch_norm(Conv3DDNNLayer(net["volume_conv2_3"],16,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    net["volume_side_op2_deconv"] = Bilinear_3DInterpolation(net["volume_side_op2"], upscale_factor=2, untie_biases=False, nonlinearity=None, pad='same')
                                                    
    #---------------------------
    net["volume_conv3_1"] = batch_norm(Conv3DDNNLayer(net["volume_pool2"],160,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv3_2"] = batch_norm(Conv3DDNNLayer(net["volume_conv3_1"],160,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv3_3"] = batch_norm(Conv3DDNNLayer(net["volume_conv3_2"],160,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same') )

    ##pool3 = Pool3DDNNLayer(conv3_3, (2,2,2), stride=2)  
    net["volume_side_op3"] = batch_norm(Conv3DDNNLayer(net["volume_conv3_3"],16,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    net["volume_side_op3_deconv"] = Bilinear_3DInterpolation(net["volume_side_op3"], upscale_factor=4, untie_biases=False, nonlinearity=None, pad='same')
    
    if params_volume.__use_newLayerAPI_dilatConv:
        #---------------------------
        net["volume_conv3_3_pad"] = PadLayer(net["volume_conv3_3"], width=2, val=0, batch_ndim=2)
        net["volume_conv4_1"] = batch_norm(DilatedConv3DLayer(net["volume_conv3_3_pad"],300,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
        net["volume_conv4_1_pad"] = PadLayer(net["volume_conv4_1"], width=2, val=0, batch_ndim=2)
        net["volume_conv4_2"] = batch_norm(DilatedConv3DLayer(net["volume_conv4_1_pad"],300,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
        net["volume_conv4_2_pad"] = PadLayer(net["volume_conv4_2"], width=2, val=0, batch_ndim=2)
        net["volume_conv4_3"] = batch_norm(DilatedConv3DLayer(net["volume_conv4_2_pad"],300,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False) )
        net["volume_conv4_3_pad"] = PadLayer(net["volume_conv4_3"], width=0, val=0, batch_ndim=2)
        net["volume_side_op4"] = batch_norm(DilatedConv3DLayer(net["volume_conv4_3_pad"],16,(1,1,1),dilation=(2,2,2),nonlinearity=nonlinearity_sigmoid,untie_biases=False))
        net["volume_side_op4_deconv"] = Bilinear_3DInterpolation(net["volume_side_op4"], upscale_factor=4, untie_biases=False, nonlinearity=None, pad='same')
    else:
        net["volume_conv4_1"] = batch_norm(Dilated_Conv3DDNNLayer(net["volume_conv3_3"],300,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
        net["volume_conv4_2"] = batch_norm(Dilated_Conv3DDNNLayer(net["volume_conv4_1"],300,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
        net["volume_conv4_3"] = batch_norm(Dilated_Conv3DDNNLayer(net["volume_conv4_2"],300,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same') )
        net["volume_side_op4"] = batch_norm(Dilated_Conv3DDNNLayer(net["volume_conv4_3"],16,(1,1,1),dilation_size=2,nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
        net["volume_side_op4_deconv"] = Bilinear_3DInterpolation(net["volume_side_op4"], upscale_factor=4, untie_biases=False, nonlinearity=None, pad='same')
                                
    #---------------------------
    net["volume_fuse_side_outputs"] = ConcatLayer([net["volume_side_op1_deconv"],net["volume_side_op2_deconv"],net["volume_side_op3_deconv"],net["volume_side_op4_deconv"]], axis=1)
    net["volume_fusion_conv"] = batch_norm(Conv3DDNNLayer(net["volume_fuse_side_outputs"],100,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_fusion_conv"] = batch_norm(Conv3DDNNLayer(net["volume_fusion_conv"],100,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_fusion_conv3"] = batch_norm(Conv3DDNNLayer(net["volume_fusion_conv"],1,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same')) # linear output for regression
    ##fusion_DilatedConv = batch_norm(Dilated_Conv3DDNNLayer(fuse_side_outputs,100,(3,3,3),dilation_size=1,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_DilatedConv = batch_norm(Dilated_Conv3DDNNLayer(fusion_DilatedConv,100,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_DilatedConv = batch_norm(Dilated_Conv3DDNNLayer(fusion_DilatedConv,100,(3,3,3),dilation_size=4,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_conv2 = batch_norm(Dilated_Conv3DDNNLayer(fusion_DilatedConv,100,(3,3,3),dilation_size=1,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_conv3 = batch_norm(Dilated_Conv3DDNNLayer(fusion_conv2,1,(3,3,3),dilation_size=1,nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same')    )
    net["output_volumeNet"] = net["volume_fusion_conv3"]
    return net



def volume_net_pureConv(input_var_5D, input_var_shape,\
        N_predicts_perGroup = 6):
    """
    from the 5D input (N_cubePair, 2rgb, h, w, d) of the colored cubePairs 
    to predicts occupancy probability output (N_cubePair, 1, h, w, d)
    """
    input_var = input_var_5D
    N_colorChannel = input_var_shape.shape(1) / 2
    net={}
    net["volume_input"] = lasagne.layers.InputLayer(input_var_shape, input_var)
    input_chunk_len = input_var.shape[0] / N_predicts_perGroup

    conv_nonlinearity = lasagne.nonlinearities.rectify
    nonlinearity_sigmoid = lasagne.nonlinearities.sigmoid

    #---------------------------    
    net["volume_merge"] = ReshapeLayer(net["volume_input"], (-1, N_colorChannel, [2], [3], [4])) # (None,N_color*2,D,D,D) ==> (None*2,N_color,D,D,D)
    net["volume_conv0_1"] = batch_norm(Conv3DDNNLayer(net["volume_merge"], 30,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv0_2"] = batch_norm(Conv3DDNNLayer(net["volume_conv0_1"],30,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_split"] = ReshapeLayer(net["volume_conv0_2"], (-1, 60, [2], [3], [4])) # (None*2,c,d,d,d) ==> (None,c*2,d,d,d)
    #---------------------------    
    net["volume_conv1_1"] = batch_norm(Conv3DDNNLayer(net["volume_split"], 90,(3,3,3), nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv1_2"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_1"], 90,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))

    # net["volume_pool1"] = Pool3DDNNLayer(net["volume_conv1_3"], (2,2,2), stride=2)
    net["volume_side_op1"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_2"],16,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    net["volume_side_op1_deconv"] = net["volume_side_op1"]

    #---------------------------
    net["volume_conv2_1"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_2"],120,(3,3,3), stride=2, nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv2_2"] = batch_norm(Conv3DDNNLayer(net["volume_conv2_1"],120,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))

    # net["volume_pool2"] = Pool3DDNNLayer(net["volume_conv2_3"], (2,2,2), stride=2)  
    net["volume_side_op2"] = batch_norm(Conv3DDNNLayer(net["volume_conv2_2"],32,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    net["volume_side_op2_deconv"] = Bilinear_3DInterpolation(net["volume_side_op2"], upscale_factor=2, untie_biases=False, nonlinearity=None, pad='same')
                                                    
    #---------------------------
    net["volume_conv3_1"] = batch_norm(Conv3DDNNLayer(net["volume_conv2_2"],150,(3,3,3), stride=2, nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv3_2"] = batch_norm(Conv3DDNNLayer(net["volume_conv3_1"],150,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same') )

    ##pool3 = Pool3DDNNLayer(conv3_3, (2,2,2), stride=2)  
    net["volume_side_op3"] = batch_norm(Conv3DDNNLayer(net["volume_conv3_2"],64,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    net["volume_side_op3_deconv"] = Bilinear_3DInterpolation(net["volume_side_op3"], upscale_factor=4, untie_biases=False, nonlinearity=None, pad='same')
    
    # if params_volume.__use_newLayerAPI_dilatConv:
    #     #---------------------------
    #     net["volume_conv3_2_pad"] = PadLayer(net["volume_conv3_2"], width=2, val=0, batch_ndim=2)
    #     net["volume_conv4_1"] = batch_norm(DilatedConv3DLayer(net["volume_conv3_2_pad"],320,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    #     net["volume_conv4_1_pad"] = PadLayer(net["volume_conv4_1"], width=2, val=0, batch_ndim=2)
    #     net["volume_conv4_2"] = batch_norm(DilatedConv3DLayer(net["volume_conv4_1_pad"],320,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    #     net["volume_conv4_2_pad"] = PadLayer(net["volume_conv4_2"], width=2, val=0, batch_ndim=2)
    #     net["volume_conv4_3"] = batch_norm(DilatedConv3DLayer(net["volume_conv4_2_pad"],320,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False) )
    #     net["volume_conv4_3_pad"] = PadLayer(net["volume_conv4_3"], width=0, val=0, batch_ndim=2)
    #     net["volume_side_op4"] = batch_norm(DilatedConv3DLayer(net["volume_conv4_3_pad"],64,(1,1,1),dilation=(2,2,2),nonlinearity=nonlinearity_sigmoid,untie_biases=False))
    #     net["volume_side_op4_deconv"] = Bilinear_3DInterpolation(net["volume_side_op4"], upscale_factor=4, untie_biases=False, nonlinearity=None, pad='same')
    # else:
    #     net["volume_conv4_1"] = batch_norm(Dilated_Conv3DDNNLayer(net["volume_conv3_2"],320,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    #     net["volume_conv4_2"] = batch_norm(Dilated_Conv3DDNNLayer(net["volume_conv4_1"],320,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    #     net["volume_conv4_3"] = batch_norm(Dilated_Conv3DDNNLayer(net["volume_conv4_2"],320,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same') )
    #     net["volume_side_op4"] = batch_norm(Dilated_Conv3DDNNLayer(net["volume_conv4_3"],64,(1,1,1),dilation_size=2,nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    #     net["volume_side_op4_deconv"] = Bilinear_3DInterpolation(net["volume_side_op4"], upscale_factor=4, untie_biases=False, nonlinearity=None, pad='same')
                                
    #---------------------------
    net["volume_fuse_side_outputs"] = ConcatLayer([net["volume_side_op1_deconv"],net["volume_side_op2_deconv"],net["volume_side_op3_deconv"]], axis=1)
    net["volume_fusion_conv"] = batch_norm(Conv3DDNNLayer(net["volume_fuse_side_outputs"],100,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_fusion_conv"] = batch_norm(Conv3DDNNLayer(net["volume_fusion_conv"],50,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_fusion_conv3"] = batch_norm(Conv3DDNNLayer(net["volume_fusion_conv"],1,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same')) # linear output for regression
    ##fusion_DilatedConv = batch_norm(Dilated_Conv3DDNNLayer(fuse_side_outputs,100,(3,3,3),dilation_size=1,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_DilatedConv = batch_norm(Dilated_Conv3DDNNLayer(fusion_DilatedConv,100,(3,3,3),dilation_size=2,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_DilatedConv = batch_norm(Dilated_Conv3DDNNLayer(fusion_DilatedConv,100,(3,3,3),dilation_size=4,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_conv2 = batch_norm(Dilated_Conv3DDNNLayer(fusion_DilatedConv,100,(3,3,3),dilation_size=1,nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    ##fusion_conv3 = batch_norm(Dilated_Conv3DDNNLayer(fusion_conv2,1,(3,3,3),dilation_size=1,nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same')    )
    net["output_volumeNet"] = net["volume_fusion_conv3"]
    return net

def volume_net_resBlock(input_var_5D, input_var_shape,\
        N_predicts_perGroup = 6):
    """
    from the 5D input (N_cubePair, 2rgb, h, w, d) of the colored cubePairs 
    to predicts occupancy probability output (N_cubePair, 1, h, w, d)
    """
    input_var = input_var_5D
    net={}
    net["volume_input"] = lasagne.layers.InputLayer(input_var_shape, input_var)
    input_chunk_len = input_var.shape[0] / N_predicts_perGroup

    conv_nonlinearity = lasagne.nonlinearities.rectify
    nonlinearity_sigmoid = lasagne.nonlinearities.sigmoid

    #---------------------------    
    net["volume_conv1_1"] = batch_norm(Conv3DDNNLayer(net["volume_input"], 50,(3,3,3), nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv1_2"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_1"], 50,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv1_3"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_2"], 50,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["side1"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_1"], 50,(1,1,1),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["add1"] = ElemwiseMergeLayer([net["side1"], net["volume_conv1_3"]], merge_function=T.add)

    #---------------------------    
    net["volume_conv1_2_pad"] = PadLayer(net["add1"], width=2, val=0, batch_ndim=2)
    net["volume_conv2_1"] = batch_norm(DilatedConv3DLayer(net["volume_conv1_2_pad"],50,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    net["volume_conv2_1_pad"] = PadLayer(net["volume_conv2_1"], width=2, val=0, batch_ndim=2)
    net["volume_conv2_2"] = batch_norm(DilatedConv3DLayer(net["volume_conv2_1_pad"],50,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    net["side2"] = batch_norm(Conv3DDNNLayer(net["volume_conv2_1"], 50,(1,1,1),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["add2"] = ElemwiseMergeLayer([net["side2"], net["volume_conv2_2"]], merge_function=T.add)

    #---------------------------    
    net["volume_conv2_3_pad"] = PadLayer(net["add2"], width=2, val=0, batch_ndim=2)
    net["volume_conv3_1"] = batch_norm(DilatedConv3DLayer(net["volume_conv2_3_pad"],50,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    net["volume_conv3_1_pad"] = PadLayer(net["volume_conv3_1"], width=2, val=0, batch_ndim=2)
    net["volume_conv3_2"] = batch_norm(DilatedConv3DLayer(net["volume_conv3_1_pad"],50,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    net["side3"] = batch_norm(Conv3DDNNLayer(net["volume_conv3_1"], 50,(1,1,1),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["add3"] = ElemwiseMergeLayer([net["side3"], net["volume_conv3_2"]], merge_function=T.add)

    # net["volume_conv3_2_pad"] = PadLayer(net["volume_conv3_2"], width=2, val=0, batch_ndim=2)
    # net["volume_conv3_3"] = batch_norm(DilatedConv3DLayer(net["volume_conv3_2_pad"],50,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False) )
    net["volume_fuse_side_outputs"] = ConcatLayer([net["add2"], net["add3"]], axis=1)
    net["volume_fusion_conv1"] = batch_norm(Conv3DDNNLayer(net["volume_fuse_side_outputs"], 50,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same')) # linear output for regression
    net["volume_fusion_conv3"] = batch_norm(Conv3DDNNLayer(net["volume_fusion_conv1"], 1,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same')) # linear output for regression
    net["output_volumeNet"] = net["volume_fusion_conv3"]
    return net

def volume_net_split_resBlock(input_var_5D, input_var_shape,\
        N_predicts_perGroup = 6):
    """
    from the 5D input (N_cubePair, 2rgb, h, w, d) of the colored cubePairs 
    to predicts occupancy probability output (N_cubePair, 1, h, w, d)
    """
    input_var = input_var_5D
    net={}
    net["volume_input"] = lasagne.layers.InputLayer(input_var_shape, input_var)
    input_chunk_len = input_var.shape[0] / N_predicts_perGroup

    conv_nonlinearity = lasagne.nonlinearities.rectify
    nonlinearity_sigmoid = lasagne.nonlinearities.sigmoid

    #---------------------------    
    net["volume_merge"] = ReshapeLayer(net["volume_input"], (-1, 1, [2], [3], [4])) # (None,2,D,D,D) ==> (None*2,1,D,D,D)
    net["volume_conv0_1"] = batch_norm(Conv3DDNNLayer(net["volume_merge"], 20,(7,7,7),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv0_2"] = batch_norm(Conv3DDNNLayer(net["volume_conv0_1"], 30,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["side0"] = batch_norm(Conv3DDNNLayer(net["volume_conv0_1"], 30,(1,1,1),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["add0"] = ElemwiseMergeLayer([net["side0"], net["volume_conv0_2"]], merge_function=T.add)

    net["volume_split"] = ReshapeLayer(net["add0"], (-1, 60, [2], [3], [4])) # (None*2,c,d,d,d) ==> (None,c*2,d,d,d)

    #---------------------------    
    net["volume_conv1_1"] = batch_norm(Conv3DDNNLayer(net["volume_split"], 50,(3,3,3), nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["volume_conv1_2"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_1"], 50,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["side1"] = batch_norm(Conv3DDNNLayer(net["volume_conv1_1"], 50,(1,1,1),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["add1"] = ElemwiseMergeLayer([net["side1"], net["volume_conv1_2"]], merge_function=T.add)

    #---------------------------    
    net["volume_conv1_2_pad"] = PadLayer(net["add1"], width=2, val=0, batch_ndim=2)
    net["volume_conv2_1"] = batch_norm(DilatedConv3DLayer(net["volume_conv1_2_pad"],50,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    net["volume_conv2_1_pad"] = PadLayer(net["volume_conv2_1"], width=2, val=0, batch_ndim=2)
    net["volume_conv2_2"] = batch_norm(DilatedConv3DLayer(net["volume_conv2_1_pad"],50,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    net["side2"] = batch_norm(Conv3DDNNLayer(net["volume_conv2_1"], 50,(1,1,1),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["add2"] = ElemwiseMergeLayer([net["side2"], net["volume_conv2_2"]], merge_function=T.add)

    #---------------------------    
    net["volume_conv2_3_pad"] = PadLayer(net["add2"], width=2, val=0, batch_ndim=2)
    net["volume_conv3_1"] = batch_norm(DilatedConv3DLayer(net["volume_conv2_3_pad"],50,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    net["volume_conv3_1_pad"] = PadLayer(net["volume_conv3_1"], width=2, val=0, batch_ndim=2)
    net["volume_conv3_2"] = batch_norm(DilatedConv3DLayer(net["volume_conv3_1_pad"],50,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    net["side3"] = batch_norm(Conv3DDNNLayer(net["volume_conv3_1"], 50,(1,1,1),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["add3"] = ElemwiseMergeLayer([net["side3"], net["volume_conv3_2"]], merge_function=T.add)

    # net["volume_conv3_2_pad"] = PadLayer(net["volume_conv3_2"], width=2, val=0, batch_ndim=2)
    # net["volume_conv3_3"] = batch_norm(DilatedConv3DLayer(net["volume_conv3_2_pad"],50,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False) )
    net["volume_fuse_side_outputs"] = ConcatLayer([net["add2"], net["add3"]], axis=1)
    net["volume_fusion_conv1"] = batch_norm(Conv3DDNNLayer(net["volume_fuse_side_outputs"], 50,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same')) # linear output for regression
    net["volume_fusion_conv3"] = batch_norm(Conv3DDNNLayer(net["volume_fusion_conv1"], 1,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same')) # linear output for regression
    net["output_volumeNet"] = net["volume_fusion_conv3"]
    return net



#---------------------------
# define fuseNet



def features_2_softmaxWeights_net(feature_input_var, Dim_feature,\
        num_hidden_units, N_predicts_perGroup):
    """
    Because the softmax weights reflect relative importance. Each value will not be meaningful without compare with other members in the group. 

    from the feature input (N_cubePair, Dim_feature) for each cube pair 
    to predict the importance/weight output (N_cubePair, 1) for surface fusion
    """
    net = {}
    net["feature_input"] = lasagne.layers.InputLayer((None,Dim_feature), feature_input_var)
    net["feature_fc1"] = batch_norm(DenseLayer(net["feature_input"], num_units=num_hidden_units, nonlinearity=lasagne.nonlinearities.sigmoid))
    net["feature_linear1"] = DenseLayer(net["feature_fc1"], num_units=1, nonlinearity=None)
    net["feature_reshape"] = ReshapeLayer(net["feature_linear1"], shape=(-1, N_predicts_perGroup))
    net["feature_softmax"] = NonlinearityLayer(net["feature_reshape"], nonlinearity=lasagne.nonlinearities.softmax)

    net["output_softmaxWeights"] = net["feature_softmax"]
    return net


def fusion_net(input_var, feature_input_var, input_cube_size, N_samples_perGroup, \
             Dim_feature, num_hidden_units, with_weight=True, N_colorChannel = 1):
    """
    Because no matter train / val / test, the fusion_net (volumeNet + features_2_softmaxWeights_net) will be build and the trained model will be loaded.
    Latter, when def_net_fn, only need to feed in the defined fusion_net.

    Check
    ===========
    >> import theano.tensor as T
    >> import nets as n
    >> import lasagne
    >> tensor5D = T.TensorType('float32', (False,)*5)
    >> input_var = tensor5D('X')
    >> similFeature_var = T.matrix('similFeature')
    >> net = n.fusion_net(input_var, similFeature_var,32,3,128,100,True)
    >> param_volum = len(lasagne.layers.get_all_params(net['output_volumeNet']))
    >> param_simil = len(lasagne.layers.get_all_params(net['feature_softmax']))
    >> param_fuse = len(lasagne.layers.get_all_params(net['output_fusionNet']))
    >> param_fuse == param_volum + param_simil
    """
    net = volume_net_side(input_var_5D = input_var, input_var_shape = (None,N_colorChannel*2)+(input_cube_size,)*3, \
            N_predicts_perGroup = N_samples_perGroup)  # volume_net_side / volume_net_pureConv / volume_net_resBlock  / volume_net_split_resBlock
    net["output_volumeNet_reshape"] = ReshapeLayer(net["output_volumeNet"], shape=(-1, N_samples_perGroup)+(input_cube_size,)*3)
    if with_weight:
        softmaxWeights_net = features_2_softmaxWeights_net(feature_input_var, Dim_feature,\
                num_hidden_units, N_samples_perGroup)
        net.update(softmaxWeights_net)
        #output_softmaxWeights_var= lasagne.layers.get_output(net["output_softmaxWeights"])
        ###output_volumeNet_channelPool = ChannelPool_argmaxWeight(output_volumeNet_reshape, average_weight_tensor)
        net["output_volumeNet_channelPool"] = ChannelPool_weightedAverage([net["output_volumeNet_reshape"], net["output_softmaxWeights"]])
        ## net["output_volumeNet_channelPool"] = ChannelPool_max(net["output_volumeNet_reshape"])
    
    else:
        net["output_volumeNet_channelPool"] = ChannelPool_max(net["output_volumeNet_reshape"])

 

    net["output_fusionNet"] = net["output_volumeNet_channelPool"] ##output_volumeNet_reshape_channelPool / conv1_3
    print "output shape:", net["output_fusionNet"].output_shape
    return net


#----------------------------------------------------------------------        
def def_updates_old(net, cost, layer_range_tuple_2_update, default_lr, update_algorithm='nesterov_momentum'):
    """
    learning rate for finetuning

    Parameters
    ----------
    net: dict of layers
    cost: cost function
    layer_range_tuple_2_update: ('layerName1','layerName2')
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
            
    if update_algorithm in 'nesterov_momentum':
        layer_updates = lasagne.updates.nesterov_momentum(cost, params_trainable, learning_rate=default_lr, momentum=0.9)
    elif update_algorithm in 'sgd; stochastic gradient descent':
        layer_updates = lasagne.updates.sgd(cost, params_trainable, learning_rate=default_lr)    
    else:
        raise ValueError("the update_algorithm {} is not found".format(update_algorithm))

    return layer_updates
    
#----------------------------------------------------------------------        
def def_updates(cost, params, default_lr, update_algorithm='nesterov_momentum'):
    """
    learning rate for finetuning

    Parameters
    ----------
    cost: cost function
    params: trainable params
    default_lr: default lr
    update_algorithm: 'sgd' / 'nesterov_momentum'
    
    Returns
    -------
    updates: for train_fn
    """
       
    if update_algorithm in 'nesterov_momentum':
        layer_updates = lasagne.updates.nesterov_momentum(cost, params, learning_rate=default_lr, momentum=0.9)
    elif update_algorithm in 'sgd; stochastic gradient descent':
        layer_updates = lasagne.updates.sgd(cost, params, learning_rate=default_lr)    
    else:
        raise ValueError("the update_algorithm {} is not found".format(update_algorithm))

    return layer_updates
 



def weighted_mult_binary_crossentropy(prediction, target, w_for_1):
    return -(w_for_1 * target * T.log(prediction) + (1.0-w_for_1)*(1.0 - target) * T.log(1.0 - prediction))

def weighted_MSE(prediction, target, w_for_1):
    w_for_0 = 1.0 - w_for_1
    w_for_p = lambda p: w_for_0 + p * (w_for_1 - w_for_0) 
    return T.sqr((prediction - target)[(target > params_volume.__soft_gt_thresh).nonzero()]) * w_for_1 + \
        T.sqr((prediction - target)[(target < params_volume.__soft_gt_thresh).nonzero()]) * w_for_0

def masked_MSE(prediction, target):
    """
    only choose partial of the negative voxels to compute the lost, as [1]

    reference:
    [1]: learning to simplify: FCN for rough sketch cleanup
    """
    sample_pos = (prediction - target)[(target > 0).nonzero()]
    sample_neg = (prediction - target)[T.eq(target, 0).nonzero()]

    indexes = T_srng.permutation(n=sample_neg.shape[0], size=(1,))# # of neg voxels
    indexes_partial = indexes.squeeze()[0: sample_pos.shape[0]] # only choose neg voxels with # of pos voxels
    sample_neg_partial = sample_neg[indexes_partial]
    return sample_pos**2 + sample_neg_partial**2 # have the same size
    
def weighted_accuracy(prediction, target):
    """
    calculate the positive/negative acc
    acc = (acc_pos + acc_neg) / 2. # equally weighted
    >>> gt = T.matrix('gt')
    >>> pred = T.matrix('pred')
    >>> acc = weighted_accuracy(pred, gt)
    >>> f = theano.function([pred, gt], acc)
    >>> pred_np = np.array([[0.1,0],[0.9,1]]).astype(np.float32)
    >>> gt_np = np.array([[0,0],[0,0]]).astype(np.float32)
    >>> f(pred_np, gt_np) == 0.5
    True
    """
    pos = (target > 0).nonzero()
    neg = T.eq(target, 0).nonzero()

    accuracy_neg = lasagne.objectives.binary_accuracy(prediction[neg], target[neg])
    # when the cube is empty, (target > 0).sum() = 0
    # accuracy_pos = accuracy_neg if (target > 0).sum() == 0 else lasagne.objectives.binary_accuracy(prediction[pos], target[pos]) 
    accuracy_pos = ifelse(T.eq((target > 0).sum(), 0), accuracy_neg, lasagne.objectives.binary_accuracy(prediction[pos], target[pos]) )

    return (T.mean(accuracy_pos) + T.mean(accuracy_neg))/2.0
     
def acc_compl_1pair_nonEmpty(prediction, target, threshold = 0.5, norm='L2'):
    """
    MVS metric: average accuracy & completeness: distance between one point cloud data pair

    prediction: 1 d-dimension volume with $n$ points
    target: 1 d-dimension volume with $m$ points

    >>> gt = T.matrix('gt')
    >>> pred = T.matrix('pred')
    >>> acc_compl = acc_compl_1pair_nonEmpty(pred, gt)
    >>> f = theano.function([pred, gt], acc_compl)
    >>> pred_np = np.array([[0.1,0.6],[0.9,0]]).astype(np.float32)
    >>> gt1_np = np.array([[0,0.8],[0.8,0.4]]).astype(np.float32)
    >>> gt2_np = np.array([[0.8,0.8],[0.2,0.4]]).astype(np.float32)
    >>> f(pred_np, gt1_np) == 0
    True
    >>> f(pred_np, gt2_np) == (0.5 + 0.5)
    True
    >>> acc_compl_L1 = acc_compl_1pair_nonEmpty(pred, gt, norm='L1')
    >>> f2 = theano.function([pred, gt], acc_compl_L1)
    >>> f2(pred_np, gt2_np) == (0.5 + 0.5)
    True
    >>> gt3_np = np.array([[0,0.8],[0.2,0.4]]).astype(np.float32)
    >>> f2(pred_np, gt3_np) == (1 + 0)
    True
    """
    coord_targ = T.shape_padaxis(T.stack((target > threshold).nonzero(), axis=1).astype('float32'), axis=1)# (n, d) --> (n, 1, d)
    coord_pred = T.shape_padaxis(T.stack((prediction > threshold).nonzero(), axis=1).astype('float32'), axis=0)# (m, d) --> (1, m, d)
    coord_diff = coord_targ - coord_pred # (n,1,d) - (1,m,d) --> (n, m, d)
    if norm == 'L2':
        coord_diff_sqr = T.sqr(coord_diff) # (n, m, d) --> (n, m)
        coord_l2_sqr = T.sum(coord_diff_sqr, axis=-1, keepdims = False) # (n, m, d) --> (n, m)
        coord_norm = T.sqrt(coord_l2_sqr) # (n, m) --> (n, m)
    elif norm == 'L1':
        coord_norm = T.sum(abs(coord_diff), axis=-1, keepdims = False) # (n, m, d) --> (n, m)
    else:
        raise Warning("`acc_compl_1pair_nonEmpty` only support L1 / L2")

    acc_compl = T.mean(T.min(coord_norm, axis=0)) + T.mean(T.min(coord_norm, axis=1)) # accuracy + completeness
    return acc_compl


def acc_compl_1pair(prediction, target, threshold = 0.5, default_dist = 10, norm='L2'):
    """
    MVS metric: average accuracy & completeness: distance between one point cloud data pair
    If either one is empty: the distance will be assigned to `default_dist`

    prediction: 1 d-dimension volume with $n$ points
    target: 1 d-dimension volume with $m$ points

    >>> gt = T.matrix('gt')
    >>> pred = T.matrix('pred')
    >>> acc_compl = acc_compl_1pair(pred, gt)
    >>> f = theano.function([pred, gt], acc_compl)
    >>> pred_np = np.array([[0.1,0.6],[0.9,0]]).astype(np.float32)
    >>> pred_0_np = np.array([[0.1,0],[0,0]]).astype(np.float32)
    >>> gt_np = np.array([[1,0.8],[0.8,0.4]]).astype(np.float32)
    >>> gt_0_np = np.array([[0,0],[0.4,0]]).astype(np.float32)
    >>> f(pred_0_np, gt_0_np) == 0 # both are empty
    True
    >>> f(pred_np, gt_0_np) == 10
    True
    >>> f(pred_0_np, gt_np) == 10
    True
    >>> np.allclose(f(pred_np, gt_np), 1/3.)
    True
    """
    coord_targ_tuple = (target > threshold).nonzero()
    coord_pred_tuple = (prediction > threshold).nonzero()
    # use */+ to implement the logic operation
    n_pt_targ = coord_targ_tuple[0].size
    n_pt_pred = coord_pred_tuple[0].size
    acc_compl = ifelse(T.eq(n_pt_targ * n_pt_pred, 0),
            ifelse(T.eq(n_pt_targ + n_pt_pred, 0), 0, default_dist).astype('float32'),
            acc_compl_1pair_nonEmpty(prediction, target, norm=norm).astype('float32')) # (default_dist * (n_pt_targ + n_pt_pred)).astype('float32')
    return acc_compl 

 
def acc_compl_Npair(predictions, targets, threshold = 0.5, default_dist = 10, norm='L2', enable_backPropag = False, shared_params = None):
    """
    MVS metric: average accuracy & completeness: distance between multiple point cloud data pairs
    If either one is empty: the distance will be assigned to `default_dist`
    predictions: N d-dimension volume with $n$ points
    targets: N d-dimension volume with $m$ points
    enable_backPropag: False: pass the pred/targ to the sequences argument of `scan`, in this case, the slice of predictions will not be treated as a
    >>> gt = T.ftensor3('gt')
    >>> pred = T.ftensor3('pred')
    >>> acc_compl = acc_compl_Npair(pred, gt, enable_backPropag = False)
    >>> f = theano.function([pred, gt], acc_compl)
    >>> pred_np = np.array([[[0,0],[0,0]], [[0.1,0],[0,0]], [[0.1,0.6],[0.9,0]], [[0.1,0.6],[0.9,0]]]).astype(np.float32)
    >>> gt_np = np.array([[[0,0],[0,0]], [[0.1,0.6],[0.9,0]], [[0,0.8],[0.8,0.4]], [[0.8,0.8],[0.2,0.4]]]).astype(np.float32)
    >>> np.allclose(f(pred_np, gt_np), np.array([0, 10, 0, 1.]))
    True
    >>> acc_compl, _ = acc_compl_Npair(pred, gt, enable_backPropag = True)
    >>> f = theano.function([pred, gt], acc_compl)
    >>> np.allclose(f(pred_np, gt_np), np.array([0, 10, 0, 1.]))
    True
    """
    if enable_backPropag:
        N = targets.shape[0]
        # Generate the components of the polynomial
        # NOTE that: when calculate the Jacobian of y w.r.t. x, cannot write as `theano.scan(lambda y_i,x: T.grad(y_i,x), sequences=y, non_sequences=x)`
        # The reason is that y_i will not be a function of x anymore, while y[i] still is.
        # Should write as: `J, updates = theano.scan(lambda i, y, x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])`
        acc_compl_s, updates = theano.scan(fn=lambda _N: acc_compl_1pair(predictions[_N], targets[_N], threshold, default_dist, norm),
                outputs_info=None, sequences=T.arange(N)) #, non_sequences=shared_params, strict=True)
        # check "Using shared variables": http://deeplearning.net/software/theano/library/scan.html#guide
        # scan: pass the shared params will speed up (strict=True)
        return acc_compl_s, updates
    else:
        acc_compl_s, _ = theano.scan(fn=lambda _pred, _targ: acc_compl_1pair(_pred, _targ, threshold, default_dist, norm),
                outputs_info=None, sequences=[predictions, targets], non_sequences=None)
        return acc_compl_s

def get_trainable_params(net, layer_range_tuple_2_update):
    """
    layer_range_tuple_2_update: ('layerName1','layerName2')
            only the params within the range ('layerName1','layerName2'] will be updated
            DON'T update the 'layerName1's params

    Notes
    -------
    If multiple range of layers will be updated.
    Just updates_old.update(updates_new), because it is OrderedDict.
    """
    if layer_range_tuple_2_update is None: 
        params_trainable = lasagne.layers.get_all_params(net["output_fusionNet"], trainable=True)
    else:
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
    return params_trainable


def def_net_fn_train_val(N_samples_perGroup, default_lr, return_train_fn=True, return_val_fn=True, with_weight=True, \
            input_cube_size = params_volume.__input_hwd, \
            Dim_feature = params_volume.__similNet_features_dim, num_hidden_units = params_volume.__similNet_hidden_dim,\
            CHANNEL_MEAN = params_volume.__CHANNEL_MEAN):

    """
    This function only defines the train_fn and the val_fn while training process.
    There are 2 training process:
    1. only train the volumeNet without weight
    2. train the softmaxWeight with(out) finetuning the volumeNet

    For the val_fn when only have validation, refer to the def_net_fn_val_test.

    ===================
    >> def_net_fn_train_val(with_weight = True)
    >> def_net_fn_train_val(with_weight = False)
    """
    train_fn = None
    val_fn = None


    tensor5D = T.TensorType('float32', (False,)*5)
    input_var = tensor5D('X')
    output_var = tensor5D('Y')
    similFeature_var = T.matrix('similFeature')

    net = fusion_net(input_var, similFeature_var, input_cube_size, N_samples_perGroup,\
            Dim_feature, num_hidden_units, with_weight, N_colorChannel = params_volume.__N_colorChannel)
    if return_val_fn:
        pred_fuse_val = lasagne.layers.get_output(net["output_fusionNet"], deterministic=True)
        # accuracy_val = lasagne.objectives.binary_accuracy(pred_fuse_val, output_var) # in case soft_gt
        ## accuracy_val = weighted_accuracy(pred_fuse_val, output_var)
        accuracy_val = acc_compl_Npair(pred_fuse_val, output_var, norm='L2', enable_backPropag = False)

        # fuseNet_val_fn = theano.function([input_var, output_var], [accuracy_val,pred_fuse_val])

        val_fn_input_var_list = [input_var, similFeature_var, output_var] if with_weight\
                else [input_var, output_var]
        val_fn_output_var_list = [accuracy_val,pred_fuse_val] if with_weight\
                else [accuracy_val,pred_fuse_val]
        val_fn = theano.function(val_fn_input_var_list, val_fn_output_var_list)
    
    if return_train_fn:
        pred_fuse = lasagne.layers.get_output(net["output_fusionNet"])
        output_softmaxWeights_var= lasagne.layers.get_output(net["output_softmaxWeights"]) if with_weight \
                else None

        param_trainable = get_trainable_params(net, params_volume.__layer_range_tuple_2_update)

        # accuracy = lasagne.objectives.binary_accuracy(pred_fuse, output_var) # in case soft_gt
        ## accuracy = weighted_accuracy(pred_fuse, output_var)
        #accuracy, updates = acc_compl_Npair(pred_fuse, output_var, norm='L1', enable_backPropag = True, shared_params = param_trainable)
        accuracy = acc_compl_Npair(pred_fuse, output_var, norm='L2', enable_backPropag = False)

        #loss = weighted_MSE(pred_fuse, output_var, w_for_1 = 0.98) \
        loss = (masked_MSE(pred_fuse, output_var) if params_volume.__soft_gt else \
            weighted_mult_binary_crossentropy(pred_fuse, output_var, w_for_1 = 0.96) )\
            + regularize_layer_params(net["output_fusionNet"],l2) * 1e-4 

        aggregated_loss = lasagne.objectives.aggregate(loss)

        #if not params_volume.__layer_range_tuple_2_update is None: 
        #    updates = def_updates(net=net, cost=aggregated_loss, layer_range_tuple_2_update=params_volume.__layer_range_tuple_2_update, \
        #            default_lr=default_lr, update_algorithm='nesterov_momentum') 
        #else:
        #    params = lasagne.layers.get_all_params(net["output_fusionNet"], trainable=True)
        #    updates = lasagne.updates.nesterov_momentum(aggregated_loss, params, learning_rate=params_volume.__lr)   

        #updates.update(def_updates(aggregated_loss, param_trainable, default_lr=params_volume.__lr, update_algorithm='nesterov_momentum'))
        updates = def_updates(aggregated_loss, param_trainable, default_lr=params_volume.__lr, update_algorithm='nesterov_momentum')


        train_fn_input_var_list = [input_var, similFeature_var, output_var] if with_weight \
                else [input_var, output_var]
        train_fn_output_var_list = [loss,accuracy, pred_fuse, output_softmaxWeights_var] if with_weight \
                else [loss,accuracy, pred_fuse]

        train_fn = theano.function(train_fn_input_var_list, train_fn_output_var_list, updates=updates)
    return net, train_fn, val_fn


def def_net_fn_test(N_samples_perGroup, with_weight=True, with_groundTruth = True, return_unfused_predict = False,\
            input_cube_size = params_volume.__input_hwd, \
            Dim_feature = params_volume.__similNet_features_dim, num_hidden_units = params_volume.__similNet_hidden_dim):
    """
    this function difines 2 net_fns, which could be used in the test phase:
    1. fuseNet_calcWeight_fn: calculate softmax weight given feature input
    2. fuseNet_fn: ouput a prediction based on the colored cube pairs with(out) weighted average. (based on whether the softmax weight is available)

    when with_weight=True/False, N_samples_perGroup=1:
        fuseNet_calcWeight_fn != None, because it will be used to find the argmax softmax weight. 

        In this case, the prediction would be the output of the volumeNet(don't need to reshape anymore). 
        So that the tensor n_samples_perGroup_var, 
            which is only used to reshape from (N_group*N_sample_perGroup,1) to (N_group, N_sample_perGroup), 
            will not be treated as input.
    
    return_unfused_predict: True: also return the unfused predictions of all the view pairs.
            This return unfused predictions could be used for color fusion. 
    ==============
    >> python -c "import nets; nets.def_net_fn_test(with_weight=True, N_samples_perGroup=1)"
    >> def_net_fn_test(with_weight=False, N_samples_perGroup=1)
    >> def_net_fn_test(with_weight=True, N_samples_perGroup=2)
    >> def_net_fn_test(with_weight=False, N_samples_perGroup=2)
    """
    fuseNet_calcWeight_fn = None
    tensor5D = T.TensorType('float32', (False,)*5)
    input_var = tensor5D('X')
    output_var = tensor5D('Y')
    similFeature_var = T.matrix('similFeature')

    # This tensor is only used to reshape from (N_group*N_sample_perGroup,1) to (N_group, N_sample_perGroup)
    # so will not used when N_samples_perGroup == 1
    n_samples_perGroup_var = T.iscalar('n_samples_perGroup') # when setted as arg of theano.function, use the 'n_samples_perGroup' to pass value 

    net = fusion_net(input_var, similFeature_var, input_cube_size, n_samples_perGroup_var,
                 Dim_feature, num_hidden_units, with_weight, N_colorChannel = params_volume.__N_colorChannel)

    ##### the fuseNet_calcWeight_fn
    if with_weight == True:
        output_softmaxWeights_var= lasagne.layers.get_output(net["output_softmaxWeights"], deterministic=True)

        fuseNet_calcWeight_fn = theano.function([similFeature_var, theano.In(n_samples_perGroup_var, value=N_samples_perGroup)], \
                output_softmaxWeights_var)
             
    ##### the fuseNet_fn
    if N_samples_perGroup >= 2:
        if with_weight == True:
            similWeight_var = T.matrix('similWeight')
            
            similWeight_input_layer = lasagne.layers.InputLayer((None,N_samples_perGroup), similWeight_var)
            net["output_volumeNet_channelPool_givenWeight"] = ChannelPool_weightedAverage([net["output_volumeNet_reshape"], similWeight_input_layer])
            net["output_fusionNet"] = net["output_volumeNet_channelPool_givenWeight"]
        else:
            net["output_volumeNet_channelPool"] = ChannelPool_max(net["output_volumeNet_reshape"])
            net["output_fusionNet"] = net["output_volumeNet_channelPool"]

        output_fusionNet_var, unfused_predictions_var = lasagne.layers.get_output([net["output_fusionNet"], net["output_volumeNet_reshape"]], \
                deterministic=True)
    elif N_samples_perGroup == 1: # if only use 1 colored Cube pair, we don't need weight any more.
        with_weight = False # IMPORTANT, in this case, the vars related to weights will be ignored
        output_fusionNet_var = lasagne.layers.get_output(net["output_volumeNet"], deterministic=True) 
        unfused_predictions_var = output_fusionNet_var

    if with_groundTruth:
        # accuracy_val_givenWeight = lasagne.objectives.binary_accuracy(output_fusionNet_var, output_var) # in case of soft_gt
        ## accuracy_val_givenWeight = weighted_accuracy(output_fusionNet_var, output_var)
        accuracy_val_givenWeight = acc_compl_Npair(output_fusionNet_var, output_var, norm='L2', enable_backPropag = False)


    # *********************
    fuseNet_fn_input_var_list = [input_var, similWeight_var] if with_weight \
            else [input_var] 
    # in the reconstruction procedure, we don't have ground truth
    fuseNet_fn_input_var_list += [output_var] if with_groundTruth else []
    # Tensor n_samples_perGroup_var is only used to reshape from (N_group*N_sample_perGroup,1) to (N_group, N_sample_perGroup)
    # so only used when N_samples_perGroup >= 2
    fuseNet_fn_input_var_list += [theano.In(n_samples_perGroup_var, value=N_samples_perGroup)] if N_samples_perGroup >= 2 \
            else [] 
    # *********************    
    if return_unfused_predict:
        fuseNet_fn_output_var_list = [accuracy_val_givenWeight, output_fusionNet_var, unfused_predictions_var] if with_groundTruth \
                else [output_fusionNet_var, unfused_predictions_var]
    else:
        fuseNet_fn_output_var_list = [accuracy_val_givenWeight, output_fusionNet_var] if with_groundTruth \
                else output_fusionNet_var
    
    # *********************
    fuseNet_fn = theano.function(fuseNet_fn_input_var_list, fuseNet_fn_output_var_list)
    return net, fuseNet_calcWeight_fn, fuseNet_fn


if __name__ == '__main__':
    import doctest
    doctest.testmod()

