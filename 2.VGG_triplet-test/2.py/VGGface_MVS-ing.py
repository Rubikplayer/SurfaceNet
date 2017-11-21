import cPickle as pickle
import lasagne
from lasagne.layers import *
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer # when gpu0
#from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import *
from lasagne.utils import *
import ipdb as pdb
from scipy import misc
import time
import struct
import random
import sys
import os
from sklearn.linear_model import LogisticRegression
from Voxel_in_Hull import voxel_in_hull
from Voxel_in_Hull import save_pcd

sys.path.append("../../1.VGG_triplet-train/3.Train_2views_triplet")
import similarityNet
sys.path.append("../../3.2D_2_3D-train/2.2D_2_3D-train")
import train_val_data as train_val_data_3_2
sys.path.append("../../4.2D_2_3D-test/2.py-reconstruction")
import train_val_data_4Test
sys.path.append("../../1.VGG_triplet-train/2.vgg_prepare_py")
import prepare_data
np.random.seed(201601)
random.seed(201603)


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
    
# def build_model(input_var):
#     net = {}
#     net['input'] = InputLayer((None, 3, hw_size, hw_size),\
#                               input_var=input_var - MEAN_IMAGE_BGR[None,:,None,None])
#     net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
#     net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
#     net['pool1'] = PoolLayer(net['conv1_2'], 2)
#     net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
#     net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
#     net['pool2'] = PoolLayer(net['conv2_2'], 2)
#     net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
#     net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
#     net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
#     net['pool3'] = PoolLayer(net['conv3_3'], 2)
#     net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
#     net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
#     net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
#     net['pool4'] = PoolLayer(net['conv4_3'], 2)
#     net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
#     net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
#     net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
#     net['pool5'] = PoolLayer(net['conv5_3'], 2) # keep the output layer in lower dim
    
#     # net['concat'] = ConcatLayer([FlattenLayer(net['pool5'], 2), 
#     #                             Get_center_2x2(net['pool1']),
#     #                             Get_center_2x2(net['pool2']),
#     #                             Get_center_2x2(net['pool3']),
#     #                             Get_center_2x2(net['pool4'])
#     #                             ], axis=1)
#     # net['L2_norm'] = Norm_L2_Layer(net['concat'])    
#     # net['feature'] = DenseLayer(net['L2_norm'], num_units=1000, nonlinearity=None)

#     net['fc1'] = lasagne.layers.batch_norm(DenseLayer(net['pool5'], num_units=600, nonlinearity=lasagne.nonlinearities.tanh))
#     net['feature'] = DenseLayer(net['fc1'], num_units=128, nonlinearity=None) # this is a linear layer

#     return net


def data_augment(img_stack,crop_size, rand_mirror = True):
    stack_shape = img_stack.shape
    rand_h_start = random.randint(0, stack_shape[2]-crop_size)
    rand_w_start = random.randint(0, stack_shape[3]-crop_size)
    img_stack = img_stack[:,:,rand_h_start:rand_h_start+crop_size,\
                          rand_w_start:rand_w_start+crop_size]
    if rand_mirror:
        selected_indx = random.sample(range(stack_shape[0]), stack_shape[0]/2)
        img_stack[selected_indx] = img_stack[selected_indx,:,:,::-1]    
    return img_stack


def iterate_triplet_minibatches(inputs, batchsize, NOofSamples = 3):
    #batchsize = batchsize * 3
    #NOofSamples = 3 # because of triplet
    end_idx = 0    
    NOofTriplets = inputs.shape[0] / NOofSamples
    for start_idx in range(0, NOofTriplets - batchsize, batchsize): # don't + 1 ???
        end_idx = start_idx
        inputs_batch = inputs[NOofSamples * start_idx: NOofSamples * (start_idx + batchsize)]
        yield data_augment(inputs_batch,crop_size=hw_size, rand_mirror=True)
    end_idx += batchsize
    inputs_batch = inputs[NOofSamples * end_idx: ]   
    yield data_augment(inputs_batch,crop_size=hw_size, rand_mirror=True)

def save_entire_model(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    # filename = os.path.join('./', filename)
    # filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(model_folder+filename, 'wb') as f:
        pickle.dump(data, f)
        print("save model to: ***** {}{} *****".format(model_folder, filename))
def print_Euclid_diff_histogram(diff_pos_Euclid_stack, diff_neg_Euclid_stack, bins=20):
    histo_range = (diff_pos_Euclid_stack.min(), diff_neg_Euclid_stack.max())
    histo_pos = np.histogram(diff_pos_Euclid_stack, bins=20, range=histo_range) # range=(0,0.001))
    histo_neg = np.histogram(diff_neg_Euclid_stack, bins=20, range=histo_range)
    histo_dist_diff_batch = np.histogram(dist_diff_stack, bins=20)
    print('||Xanchor-Xpos||_2  counter: \n{}'\
          .format(histo_pos[0])) #[0]
    print('||Xanchor-Xneg||_2  counter: \n{}'\
          .format(histo_neg[0]))  
    print('||Xanchor-X_neg/pos||_2  bins edge: \n{}'\
                  .format(histo_neg[1]))         
    print('distance difference  counter: \n{}'\
                  .format(histo_dist_diff_batch))             

def calculate_Euclid_diff_threshold(diff_pos_Euclid_stack, diff_neg_Euclid_stack):
    x1 = diff_pos_Euclid_stack.reshape(-1,1) #make sure it's 2D numpy array
    x2 = diff_neg_Euclid_stack.reshape(-1,1)
    y1 = np.zeros(x1.size)
    y2 = np.ones(x2.size)
    X = np.concatenate((x1,x2),axis=0)
    
    classifier = LogisticRegression()
    classifier.fit(X-X.mean(),np.concatenate((y1,y2)))
    
    threshold = classifier.intercept_ / classifier.coef_ * -1 + X.mean()
    #print('Euclid_diff_threshold = {}'.format(threshold))
    return threshold

def load_dataset_list(sampleIndx_list, mode_list=['hard']):
    # 5D inputs is like: (len(samplefile),NOofViews,3,32,32)
    # 2D targets is like: (len(samplefile),NOofViews + 1)
    inputs_stack = np.array([])
    for mode in mode_list:
        for sampleIndx in sampleIndx_list:
            inputs = np.load(open( dataset_folder+str(sampleIndx).zfill(3)+'_'+mode+'_inputs.data'))
            inputs_stack = np.vstack([inputs_stack, inputs]) if inputs_stack.size else inputs
            print ("loaded: "+dataset_folder + str(sampleIndx).zfill(3)+'_'+mode+'_inputs.data')
    return inputs_stack.reshape(-1, inputs_stack.shape[-3], inputs_stack.shape[-2], inputs_stack.shape[-1])


def get_rgba_batch(inputs, labels):
    ## inputs: (N*NoofView, Channel, Height, Width)
    ## visibility: (N)
    pt_color = inputs[::2, :, inputs_batch.shape[-2]/2, inputs_batch.shape[-2]/2]
    BGR = pt_color.astype(np.uint8)
    #A = (on_off_probab*255).astype(np.uint8)
    A = np.asarray(254, dtype=np.uint8)
    N = labels.shape[0]
    rgba = np.zeros((N,),dtype=np.float32)
    for i in range(0, N):
        # thanks to: 'http://www.pcl-users.org/How-to-convert-from-RGB-to-float-td4022278.html'
        rgba[i] = struct.unpack('f', chr(BGR[i,0]) + chr(BGR[i,1]) + chr(BGR[i,2]) + chr(A))[0]
    return  rgba



train_on = False
val_on = False
test_on = True

shuffle_sample_ON = True
save_weight_ON = True

use_pretrained_model = True
use_VGG16_weights = False

only_train_last_layer = True
batch_size = 1500 if only_train_last_layer else 300
weight_decay = 0.1
triplet_alpha = 30 
DEFAULT_LR = 0.001

hw_size = 64
sampleIndx_list_train=[6,11,66]
sampleIndx_list_val=[3]

if test_on:
    batch_size_test = 2000
    datasetName = 'MVS' # 'MVS'/'Tsinghua'
    if datasetName == 'MVS':
        modelIndx = 17   # the model which we are reconstructing . [trained by 3/6/11/66]
        #viewIndxes = [15,16]
            #[15,16,17,23,24,25,26,32,33,34,35,36]
        #11,10,9,8]  # the views we used to reconstruct the surface
    if datasetName == 'Tsinghua':
        modelIndx = 'G'
        #viewIndxes = [0,1]   # use to determine how many views are used in one group

#dataset_folder = '/home/mengqi/dataset/MSRA-face/data_generator_orig_250/'
pcl_pcd2ply_exe = '/home/mengqi/Downloads/pcl-trunk/build/bin/pcl_pcd2ply'
dataset_folder = '/home/mengqi/dataset/MVS/lasagne/save_inputs_target_2views_triplet/'
dataset_folder_4test = '/home/mengqi/dataset/MSRA-face/data_generator_lfw_224/'
model_folder = '/home/mengqi/dataset/MVS/lasagne/save_model_2views_triplet/'
if not os.path.exists(model_folder):
        os.makedirs(model_folder)
triplet_stack_train_file = dataset_folder+'train.data'
triplet_stack_val_file = dataset_folder+'val.data'
img_stack_test_file = dataset_folder_4test+'img4D_pairs.data'

if train_on:
    triplet_stack_train = load_dataset_list(sampleIndx_list_train)
if val_on:
    triplet_stack_val = load_dataset_list(sampleIndx_list_val)
# #if test_on:
#     #img_stack_test = np.load(open(img_stack_test_file))        
#     #print('data for test is loaded: {}'.format(img_stack_test_file))

# #model = pickle.load(open('/home/mji/theano/lasagne/Recipes/examples/VGG16/vgg16.pkl'))
# #MEAN_IMAGE_BGR = model['mean value'].astype(np.float32)
# MEAN_IMAGE_BGR = np.asarray([103.939,  116.779,  123.68]).astype(np.float32)

# ############## build the network and compute the cost ##############
# input_var = T.tensor4('inputs')
# #target_var = T.bmatrix('targets')
# net = build_model(input_var)
# feature_embed = lasagne.layers.get_output(net['feature'])
# test_feature_embed = lasagne.layers.get_output(net['feature'], deterministic=True)

# def get_dist_from_feature_embed(feature_embed):
#     diff_pos = feature_embed[::3,] - feature_embed[1::3,]
#     diff_neg = feature_embed[::3,] - feature_embed[2::3,]
#     diff_pos_sq_sum = (diff_pos**2).sum(axis=1)
#     diff_neg_sq_sum = (diff_neg**2).sum(axis=1)
#     diff_pos_Euclid = diff_pos_sq_sum ** .5
#     diff_neg_Euclid = diff_neg_sq_sum ** .5
#     dist = triplet_alpha - (diff_neg_sq_sum - diff_pos_sq_sum)
#     return [dist, diff_pos_Euclid, diff_neg_Euclid]

# dist = get_dist_from_feature_embed(feature_embed)[0]
# dist_thresh = dist*(dist>0)

# if test_on:
#     diff_X = feature_embed[::2,] - feature_embed[1::2,]
#     diff_X_sq_sum = (diff_X**2).sum(axis=1)
#     diff_X_Euclid = diff_X_sq_sum ** .5
    
# ############## cost with regularization ##############
# weight_l2_penalty = lasagne.regularization.regularize_network_params(net['feature'], lasagne.regularization.l2) * weight_decay
# cost = dist_thresh.sum() + weight_l2_penalty

# ############## learning rate for finetuning ##############
# lr_mult = {}
# lr_mult['feature'] = 1
# updates = {}
# if only_train_last_layer:
#     lr_coeff_of_pre_layers = 0
# else:       
#     lr_coeff_of_pre_layers = 0.01
# print('lr_coefficient of previous layers: {}'.format(lr_coeff_of_pre_layers))
# for name, layer in net.items():
#     layer_params = layer.get_params(trainable=True)
#     layer_lr = lr_mult.get(name, lr_coeff_of_pre_layers) * DEFAULT_LR # dict.get() if name is not in the dict, get the default value
#     #layer_updates = lasagne.updates.nesterov_momentum(cost, layer_params, \
#                                     #learning_rate=layer_lr, momentum=0.9)
#     layer_updates = lasagne.updates.sgd(cost, layer_params, learning_rate=layer_lr)    
#     updates.update(layer_updates)   
 
############## train/val/test functions ##############    
#check_weight_sum = T.sum(lasagne.layers.get_output(net['fc7'],deterministic=True)) # used to check the weights befor 'fc6' don't change
#check_weight_sum_feature = T.sum(lasagne.layers.get_output(net['feature'],deterministic=True))
if train_on:
    train_fn = theano.function([input_var], [cost,feature_embed]+
                               get_dist_from_feature_embed(feature_embed), updates=updates)    
if val_on:
    val_fn = theano.function([input_var], get_dist_from_feature_embed(test_feature_embed))    
if test_on:
    # in this case the input_var is in group of 2 rather than 3
    ## test_fn = theano.function([input_var], feature_embed)  
    net_featureEmbed_layer,net_featurePair2simil_similarity_layer,patch2feature_fn,featurePair2simil_fn = \
            similarityNet.def_patch_TO_feature_TO_similarity_net_fn()
############## load the pretrained model ##############  
if use_pretrained_model:
    if use_VGG16_weights:
        filename = '/home/mji/theano/lasagne/Recipes/examples/VGG16/vgg16.pkl'
    else:
        #filename = '/home/mengqi/dataset/MSRA-face/save_model/20160201-01-vggface-model/epo19.model'
        #filename = '/home/mengqi/dataset/MSRA-face/save_model/20160202-01-2.1-vggface/epoch3.model'
        #filename = '/home/mengqi/dataset/MSRA-face/save_model/20160202-01-2.1-vggface-tune_previous_layers/epoch1.model'
        #filename = '/home/mengqi/dataset/MVS/lasagne/save_model_2views_triplet/20160303-1/epoch13.model'
        ## filename = '/home/mengqi/dataset/MVS/lasagne/save_inputs_target_2views_triplet/models/epoch5_thresh729.234.model'
        # filename = '/home/mengqi/dataset/MVS/lasagne/save_model_2views_triplet/epoch3_acc_tr0.719_val0.646.model'
        filename = '/home/mengqi/dataset/MVS/lasagne/save_model_2views_triplet/epoch7_acc_tr0.795_val0.789.model'
        # Euclid_threshold_4_test = 729.234
    with open(filename) as f:
        data = pickle.load(f)
        if use_VGG16_weights:
            data = data['param values']
            lasagne.layers.set_all_param_values(net['conv5_3'], data[:-6])
        else:
            lasagne.layers.set_all_param_values([net_featureEmbed_layer,net_featurePair2simil_similarity_layer], data)
        print('loaded the weight: {}'.format(filename))

print("learning rate: {}, batch size: {}, triplet_alpha: {}, shuffle_sample_ON: {}, weight_decay: {}"\
      .format(DEFAULT_LR, batch_size, triplet_alpha, shuffle_sample_ON, weight_decay))
for epoch in xrange(50):
    
    print("============ epoch {} ===========".format(epoch))
    
    if train_on:
        # In each epoch, we do a full pass over the training data:
        train_cost = 0
        #train_acc = 0
        train_batches = 0
        diff_pos_Euclid_stack = np.array([])
        diff_neg_Euclid_stack = np.array([])
        dist_diff_stack = np.array([])
        if shuffle_sample_ON:
            # shuffle in group of 3:
            np.random.shuffle(triplet_stack_train.reshape(triplet_stack_train.shape[0]/3, 3, 3, \
                                    triplet_stack_train.shape[-2], triplet_stack_train.shape[-2]))
        for batch in iterate_triplet_minibatches(triplet_stack_train, batch_size): # [:4503] [:900]
            output = train_fn(batch)
            train_cost += output[0]
            diff_pos_Euclid_batch = output[-2]
            diff_neg_Euclid_batch = output[-1]
            dist_diff_batch = output[-3]
            # print('current train cost:  '+str(train_err))
            train_batches += 1
            #train_acc += acc
            diff_pos_Euclid_stack = np.concatenate([diff_pos_Euclid_stack, diff_pos_Euclid_batch]) 
            diff_neg_Euclid_stack = np.concatenate([diff_neg_Euclid_stack, diff_neg_Euclid_batch]) 
            dist_diff_stack = np.concatenate([dist_diff_stack, dist_diff_batch]) 
            
        print("training loss:\t\t{:.6f}".format(train_cost / diff_pos_Euclid_stack.shape[0]))
        print_Euclid_diff_histogram(diff_pos_Euclid_stack, diff_neg_Euclid_stack)
        Euclid_diff_threshold = calculate_Euclid_diff_threshold(diff_pos_Euclid_stack, diff_neg_Euclid_stack)
        acc_train = np.concatenate((diff_pos_Euclid_stack<Euclid_diff_threshold,
                                    diff_neg_Euclid_stack>Euclid_diff_threshold), 
                                     axis=0).sum().astype(np.float32)/(diff_pos_Euclid_stack.size+diff_pos_Euclid_stack.size)
        
        if save_weight_ON:
            if (epoch+1) % 2 == 0:
                save_entire_model(net['feature'], 'epoch{}.model'.format(epoch))            
        
    if val_on:
        # In each epoch, we do a full pass over the validation data:
        val_cost = 0
        #val_acc = 0
        val_batches = 0
        diff_pos_Euclid_stack = np.array([])
        diff_neg_Euclid_stack = np.array([])
        dist_diff_stack = np.array([])
        
        for batch in iterate_triplet_minibatches(triplet_stack_val, batch_size): # [:4503]
            output = val_fn(batch)
            #val_cost += output[1]
            dist_diff_batch = output[0]
            diff_pos_Euclid_batch = output[1]
            diff_neg_Euclid_batch = output[2]
            
            # print('current val cost:  '+str(val_err))
            val_batches += 1
            #val_acc += acc
            diff_pos_Euclid_stack = np.concatenate([diff_pos_Euclid_stack, diff_pos_Euclid_batch]) 
            diff_neg_Euclid_stack = np.concatenate([diff_neg_Euclid_stack, diff_neg_Euclid_batch]) 
            dist_diff_stack = np.concatenate([dist_diff_stack, dist_diff_batch]) 
            
        #print("validation loss:\t\t{:.6f}".format(val_cost / diff_pos_Euclid_stack.shape[0]))
        print_Euclid_diff_histogram(diff_pos_Euclid_stack, diff_neg_Euclid_stack)
        acc_val = np.concatenate((diff_pos_Euclid_stack<Euclid_diff_threshold,
                              diff_neg_Euclid_stack>Euclid_diff_threshold), 
                             axis=0).sum().astype(np.float32)/(diff_pos_Euclid_stack.size+diff_pos_Euclid_stack.size)
        print("train/val Accuracy = {} / {} \n with threshold: {}".format(acc_train,acc_val,Euclid_diff_threshold))
        
        #histo_pos = np.histogram(diff_pos_Euclid_stack, bins=20) #range=(0,0.001)
        #histo_neg = np.histogram(diff_neg_Euclid_stack, bins=20)        
        #histo_dist_diff_batch = np.histogram(dist_diff_stack, bins=20)
        #print('||Xanchor-Xpos||_2  counter: \n{}'\
              #.format(histo_pos)) #[0]
        #print('||Xanchor-Xneg||_2  counter: \n{}'\
              #.format(histo_neg))     
        #print('distance difference  counter: \n{}'\
              #.format(histo_dist_diff_batch))         

              
    #if test_on:
        #for img_stacks in [img_stack_test[:500*2], img_stack_test[500*2:]]:
            ## In each epoch, we do a full pass over the test data:
            ##val_acc = 0
            #test_batches = 0
            #diff_X_Euclid_stack = np.array([])
            
            #for batch in iterate_triplet_minibatches(img_stacks, batch_size, NOofSamples=2): # [:450]
                #output = test_fn(batch)
                #diff_X_Euclid_batch = output[0]
                
                #test_batches += 1
                #diff_X_Euclid_stack = np.concatenate([diff_X_Euclid_stack, diff_X_Euclid_batch]) 
                
            #print('test result:')
            #print(np.histogram(diff_X_Euclid_stack, bins=20))
        
        
    __RETURN_ALL_VIEWPAIRS = 0

    if test_on:
        start_time = time.time()

        __root_path = '/home/mengqi/dataset/MVS/'
        __camera_po_txt_fld = __root_path + 'pos/'
        __camera_T_txt_fld = __root_path + 'cameraT/'
        # feature size of the returned similNet feature matrix
        __similNet_features_dim = 1+1 # in order to save memory, don't return useless features
        __debug_ON = False 
        __similNet_batchSize = 5600 # limited by gpu memory
        __similNet_input_hw = 32
        __dissimilarity_threshold = 0.5
        __percent_of_N_valid_viewPairs_thresh = 1/5.
        __resolution = train_val_data_4Test.resol
        __stride_ratio = train_val_data_4Test.cube_stride_over_Dsize
        __view_set = range(1,50)

        pt_batch_N = 100 if __debug_ON else 2000  # limited by pc memory

        cameraTs = prepare_data.load_cameraT_as_np(__camera_T_txt_fld)
        cameraPOs = prepare_data.load_cameraPos_as_np(__camera_po_txt_fld)
        models_img = train_val_data_3_2.load_modellist_meanIMG([modelIndx])

        Samples_param_np, cube_Dsize = train_val_data_4Test.RECONSTR_generate_3DSamples_param_np(modelIndx)
        N_pts_orig = Samples_param_np.shape[0]

        similNet_dissimilarity_list, all0_patchPair_indicator_list, patch_center_rgb_on_diff_viewPair_list, patchChannel_valueRange_list = [], [], [], []
        _iterations, _remainder = N_pts_orig / pt_batch_N, N_pts_orig % pt_batch_N
        for _i in range(0, _iterations+1): ## divide into multiple parts to reconstruct, because of the limited memory~
            print("start iter {} / {} ".format(_i, _iterations))
            if (_i ==  _iterations) and (_remainder != 0):
                Samples_select = range(_i*pt_batch_N, N_pts_orig)   
            elif (_i ==  _iterations) and (_remainder == 0):
                continue
            else:
                Samples_select = range(_i*pt_batch_N, (_i+1)*pt_batch_N)  

            _param = Samples_param_np[Samples_select]

            all_viewPairs_viewIndx, _similNet_features_sub, _all0_patchPair_indicator_sub, _patch_center_rgb_on_diff_viewPair_np_sub, _patchChannel_valueRange_sub= \
                    train_val_data_3_2.perform_similNet(patch2feature_fn=patch2feature_fn,featurePair2simil_fn=featurePair2simil_fn, \
                    occupiedCubes_param = _param, N_select_viewPairs = __RETURN_ALL_VIEWPAIRS, models_img=models_img, \
                    view_set = __view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=__similNet_input_hw, batch_size=__similNet_batchSize, \
                    similNet_features_dim = __similNet_features_dim, return_all0_patchPair_indicator = True, return_patch_center_rgb=True, return_patchChannel_valueRange=True)
            print("only 1 all0 patch goes through network takes {}".format(time.time() - start_time))
            # if __debug_ON: 
                # # can check the correctness of the 'perform_similNet' when return_all0_patchPair_indicator = True
                # start_time = time.time()
                # all_viewPairs_debug, _similNet_features_sub_debug = train_val_data_3_2.perform_similNet(patch2feature_fn=patch2feature_fn,featurePair2simil_fn=featurePair2simil_fn, \
                #         occupiedCubes_param = _param, N_select_viewPairs = __RETURN_ALL_VIEWPAIRS, models_img=models_img, \
                #         view_set = __view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=__similNet_input_hw, batch_size=__similNet_batchSize, \
                #         similNet_features_dim = __similNet_features_dim)
                # print("DEBUG mode: all the all0 patches go through network takes {}. The _similNet_features_sub_debug are exactly the same with _similNet_features_sub : {} the max difference is {}".format(\
                #         time.time() - start_time, np.allclose(_similNet_features_sub_debug, _similNet_features_sub), \
                #         abs(_similNet_features_sub_debug-_similNet_features_sub).max())) # use np.allclose rather than ==

            similNet_dissimilarity_list.append(_similNet_features_sub[:,:,-2])
            all0_patchPair_indicator_list.append(_all0_patchPair_indicator_sub)
            patch_center_rgb_on_diff_viewPair_list.append(_patch_center_rgb_on_diff_viewPair_np_sub)
            patchChannel_valueRange_list.append(_patchChannel_valueRange_sub) #(n_pts, N_viewPairs*2, c)
        


        similNet_dissimilarity_np = np.concatenate(similNet_dissimilarity_list) # (N_pts, N_viewPairs)
        all0_patchPair_indicator_np = np.concatenate(all0_patchPair_indicator_list) # (N_pts, N_viewPairs)
        patch_center_rgb_on_diff_viewPair_np = np.concatenate(patch_center_rgb_on_diff_viewPair_list) # (N_pts, N_viewPairs, c)
        patchChannel_valueRange_np = np.concatenate(patchChannel_valueRange_list) #(N_pts, N_viewPairs*2, c)

        ## TODO: need or not?
        ## # filter out the near identical patches (very similar patch pair), in order to filter out the pure color background
        similNet_dissimilarity_np[similNet_dissimilarity_np < 0.1] = 1
        #(N_pts, N_viewPairs*2, c) ==> (N_pts, N_viewPairs, 2, c) ==> (N_pts, N_viewPairs)
        similNet_dissimilarity_np[np.amax(patchChannel_valueRange_np.reshape((N_pts_orig, -1, 2, 3)),axis=(-1,-2)) < 20] = 1

        # ignor the view pairs whose patches are all black (most cases are invisible / out of scope)
        # if the dissimilarity is smaller than a threshold, we treat this as one on surface decision for a pt from a view pair
        masked_onSurface_proposals_np=np.ma.masked_array(similNet_dissimilarity_np < __dissimilarity_threshold, \
                mask=all0_patchPair_indicator_np)
       
        # if both patches of a view pair is full black, we treat it as invalid view pair
        N_valid_viewPairs = (1-all0_patchPair_indicator_np).sum(axis=-1) # sum along viewPairs dim
        # count how many view pairs give on surf proposals for each pt 
        N_onSurf_proposals = np.ma.getdata(masked_onSurface_proposals_np.sum(axis=-1)) # sum along viewPairs dim
        # if # of onsurf_proposals is larger than a _threshold_percentage of the max decision it could get from all view pairs
        # we treat this pt as on surface point.
        onSurf_decision = (N_onSurf_proposals > N_valid_viewPairs * __percent_of_N_valid_viewPairs_thresh).astype(np.bool)  # (N_pts,)
            
        # calculate the color: average along the view pairs which believe this pt is a on surf point.

        # the mask of the color array: mask (ignor) the all-black patch-pairs AND the view-pairs without on-surface-proposals, which is stored in the masked_onSurface_proposals_np array.
        mask_patch_center_rgb_on_diff_viewPair = (all0_patchPair_indicator_np & ~np.ma.getdata(masked_onSurface_proposals_np))
        # (N_pts, N_viewPairs) ==> (N_pts, N_viewPairs, c)
        _, mask_patch_center_rgb_on_diff_viewPair_3D = np.broadcast_arrays(patch_center_rgb_on_diff_viewPair_np, mask_patch_center_rgb_on_diff_viewPair[...,None])
        masked_patch_center_rgb_on_diff_viewPair =np.ma.masked_array(patch_center_rgb_on_diff_viewPair_np, \
                mask=mask_patch_center_rgb_on_diff_viewPair_3D)
        patch_center_rgb = np.ma.getdata(masked_patch_center_rgb_on_diff_viewPair.mean(axis=1)).astype(np.uint8) # (N_pts, N_viewPairs, c) ==> (N_pts, c)
        N_filtered_pts = onSurf_decision.sum()
        xyz_filtered_pts = Samples_param_np[onSurf_decision,:3]
        rgb_filtered_pts = patch_center_rgb[onSurf_decision]
        A = 254 
        rgba = np.zeros((N_filtered_pts,),dtype=np.float32)
        for i in range(N_filtered_pts):
            # thanks to: 'http://www.pcl-users.org/How-to-convert-from-RGB-to-float-td4022278.html'
            rgba[i] = struct.unpack('f', chr(rgb_filtered_pts[i,2]) + chr(rgb_filtered_pts[i,1]) + chr(rgb_filtered_pts[i,0]) + chr(A))[0]
        xyzrgba = np.c_[xyz_filtered_pts.astype(np.float32), rgba] # make sure this is float32 before save
        
        print("it takes {:.3f}s".format(time.time() - start_time))
        saved_pcd_file_name = '{}_model{}_thre{:.3f}_{}-{}Pts_resol{:.3f}_strideRatio{:.2f}_{:.1f}s.pcd'. \
                    format(datasetName, modelIndx, __dissimilarity_threshold, N_pts_orig, \
                    N_filtered_pts, __resolution, __stride_ratio, time.time() - start_time)
            
        train_val_data_3_2.RECONSTR_save_pcd(xyzrgba, model_folder, saved_pcd_file_name)
        os.system( "{} {} {}".format(pcl_pcd2ply_exe, model_folder+saved_pcd_file_name,\
                   (model_folder+saved_pcd_file_name).replace('.pcd','.ply')) )
        

    if not train_on:
        break

        
