""" get rid of theano/lasagne/caffe in this file """

import params_similNet as param
import lasagne
import cPickle as pickle

import math
import pdb
from scipy import misc
import time
import random
import sys
import os
from sklearn.linear_model import LogisticRegression
import itertools
import similarityNet 
import numpy as np

np.random.seed(201601)


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
    if batchsize <= NOofTriplets: 
        for start_idx in range(0, NOofTriplets - batchsize, batchsize): # don't + 1 ???
            end_idx = start_idx
            inputs_batch = inputs[NOofSamples * start_idx: NOofSamples * (start_idx + batchsize)]
            yield data_augment(inputs_batch,crop_size=param.__hw_size, rand_mirror=True)
        end_idx += batchsize
    inputs_batch = inputs[NOofSamples * end_idx: ]   
    yield data_augment(inputs_batch,crop_size=param.__hw_size, rand_mirror=True)

def save_entire_model(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    # filename = os.path.join('./', filename)
    # filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(model_folder+filename, 'wb') as f:
        pickle.dump(data, f)
        print("save model to: ***** {}{} *****".format(model_folder, filename))
            

def load_dataset_list(sampleIndx_list, mode_list=['hard']):
    # 5D inputs is like: (len(samplefile),NOofViews,3,32,32)
    # 2D targets is like: (len(samplefile),NOofViews + 1)
    inputs_stack = [] # get rid of multiple np.vstack for faster loading
    for mode in mode_list:
        for sampleIndx in sampleIndx_list:
            inputs = np.load(open( dataset_folder+str(sampleIndx).zfill(3)+'_'+mode+'_inputs.data'))
            #inputs_stack = np.vstack([inputs_stack, inputs]) if inputs_stack.size else inputs
            inputs_stack.append(inputs)
            print ("loaded: "+dataset_folder + str(sampleIndx).zfill(3)+'_'+mode+'_inputs.data')
    inputs_stack = np.vstack(inputs_stack) 
    return inputs_stack.reshape(-1, inputs_stack.shape[-3], inputs_stack.shape[-2], inputs_stack.shape[-1])

def filterOutNearPurePatches_4D(array4D):
    # 4D array (N*3,C,H,W)
    # 5D array (N,3,C,H,W), the 3 means triplet input tuple
    array5D = array4D.reshape((array4D.shape[0]/3,3)+array4D.shape[1:])
    array5D_diff = np.amax(array5D,axis=(-1,-2))-np.amin(array5D,axis=(-1,-2))
    array5D_new = array5D[np.amin(np.amax(array5D_diff,axis=2),axis=1) > (array5D.max()/10.)]
    return array5D_new.reshape((-1,)+array4D.shape[1:])


def main():

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    net_train_val, train_fn, val_fn = similarityNet.def_net_fn_train_val(return_train_fn=param.__train_on, return_val_fn=param.__val_on)
    net_test, test_fn = similarityNet.def_net_fn_test() if param.__test_on else None, None
    ############## load the pretrained model ##############  
    if param.__use_pretrained_model:
        if param.__use_VGG16_weights:
            filename = '/home/mengqi/dataset/MVS/lasagne/save_model_2views_triplet/vgg16.pkl'
            # download vgg16.pkl:<!wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl>
            # For details: https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
        else:
            #filename = '/home/mengqi/dataset/MSRA-face/save_model/20160201-01-vggface-model/epo19.model'
            #filename = '/home/mengqi/dataset/MSRA-face/save_model/20160202-01-2.1-vggface/epoch3.model'
            #filename = '/home/mengqi/dataset/MSRA-face/save_model/20160202-01-2.1-vggface-tune_previous_layers/epoch1.model'
            #filename = '/home/mengqi/dataset/MVS/lasagne/save_model_2views_triplet/epoch5_acc_tr0.739_val0.655.model'
            #filename = '/home/mengqi/dataset/MVS/lasagne/save_model_2views_triplet/epoch1_acc_tr0.826_val0.781.model'
            filename = '/home/mengqi/dataset/MVS/lasagne/save_model_2views_triplet/epoch47_acc_tr0.872_val0.0.model'
            #filename = '/home/mengqi/dataset/MVS/lasagne/save_model_2views_triplet/epoch3_acc_tr0.719_val0.646.model'
            Euclid_diff_threshold = 7.62327
        with open(filename) as f:
            data = pickle.load(f)
            if param.__use_VGG16_weights:
                data = data['param values']
                lasagne.layers.set_all_param_values(net_train_val['conv5_3'], data[:-6])
            else:
                lasagne.layers.set_all_param_values(net_train_val['similarity'], data)
            print('loaded the weight: '+filename)
        
    print("learning rate: {}, batch size: {}/{}/{}, triplet_alpha: {}, param.__shuffle_sample_ON: {}, weight_decay: {}"\
        .format(param.__DEFAULT_LR, param.__batch_size_train, param.__batch_size_val, param.__batch_size_test, \
        param.__triplet_alpha, param.__shuffle_sample_ON, param.__weight_decay))


    train_val_acc_record = []

    for epoch in xrange(param.__train_epoch):
        print("============ epoch {} ===========".format(epoch))
        start_time = time.time()
        
        if param.__train_on:                
            # In each epoch, we do a full pass over the training data:
            train_cost = 0
            similarity_acc = np.array([])

            if param.__shuffle_sample_ON:
                # shuffle in group of 3:
                np.random.shuffle(triplet_stack_train.reshape(triplet_stack_train.shape[0]/3, 3, 3, \
                                        triplet_stack_train.shape[-2], triplet_stack_train.shape[-2]))
            for batch in iterate_triplet_minibatches(triplet_stack_train, param.__batch_size_train): # [:4503] [:209*3]
                output = train_fn(batch) 
                train_cost += output[0]
                similarity_acc = np.append(similarity_acc,output[1])
            
            acc_train = similarity_acc.mean()
            print("training loss:\t\t{:.6f}".format(train_cost / similarity_acc.shape[0]))
            if math.isnan(train_cost):
                break

        acc_val = 0.0
        if param.__val_on:
            # In each epoch, we do a full pass over the validation data:
            similarity_acc = np.array([])
            
            for batch in iterate_triplet_minibatches(triplet_stack_val, param.__batch_size_val): # [:4503]
                output = val_fn(batch)
                similarity_acc = np.append(similarity_acc,output[0])
                
            acc_val = similarity_acc.mean()
        print("train/val Accuracy = {:.5f} / {:.5f}".format(acc_train if param.__train_on else 0,acc_val))
   
        
        if param.__save_weight_ON and param.__train_on:
            if (epoch+1) % 2 == 0:
                save_entire_model(net_train_val['similarity'], 'epoch{}_acc_tr{:.3}_val{:.3}.model'.format(epoch,acc_train,acc_val))            
            
                
        if param.__test_on:
            for img_stacks in [img_stack_test[:500*2], img_stack_test[500*2:]]:
                # In each epoch, we do a full pass over the test data:
                
                for batch in iterate_triplet_minibatches(img_stacks, param.__batch_size_test, NOofSamples=2): # [:450]
                    output = test_fn(batch)
                    similarity = output[0]

        if (not param.__train_on):
            break
        
        train_val_acc_record.append((acc_train+acc_val)/2.)
        if param.__debugON and (epoch > 3) and (train_val_acc_record[-1]-np.mean(train_val_acc_record[-3:]) < 0.01): # early stop !!!
            break

    train_val_mean_acc.append(max(train_val_acc_record)) # only append the accuracy at the last epoch






__non_trainVal_set = [23,24,27,29,73,  114,118,  \
        25,26,27,  1,11,24,34,49,62,  11,32,33,48,75,\
        110,25,1,4,77,  1,9,10,12,13,15,33,54,  78,79,80,81] 
__trainVal_set = [i for i in range(1,129) if i not in __non_trainVal_set]
__rand_val_set = [35, 37, 43, 5, 66, 117, 17, 106, 21, 40, 82, 56, 86, 3, 67, 28, 38, 59] # 18 randomly selected models
__rand_train_set = [i for i in __trainVal_set if i not in __rand_val_set]



sampleIndx_list_train= [118] if param.__mini_dataset else __rand_train_set # [118] #
# [5,6,16,18,19,21,36,37,38,42,45,49,50,59,61,62,63,66,\
# 85,89,93,96,97,100,103,104,105,124,128] #if not param.__debugON else [6,11,66]
sampleIndx_list_val= [35] if param.__mini_dataset else __rand_val_set # [35] #
# [3] #if param.__debugON else [17,31,84,102,125,126] #[3]#
# sampleIndx_list_train=[5,6,16,18,19,21,36,37,38,42,45,49,50,59,61,62,63,66,\
# 85,89,93,96,97,100,103,104,105,124,128]
# sampleIndx_list_val=[17,31,84,102,125,126]

#dataset_folder = '/home/mengqi/dataset/MSRA-face/data_generator_orig_250/'
dataset_folder = '/home/mengqi/dataset/MVS/lasagne/save_inputs_target_2views_triplet/'
dataset_folder_4test = '/home/mengqi/dataset/MSRA-face/data_generator_lfw_224/'
model_folder = '/home/mengqi/dataset/MVS/lasagne/save_model_2views_triplet/'



triplet_stack_train_file = dataset_folder+'train.data'
triplet_stack_val_file = dataset_folder+'val.data'
img_stack_test_file = dataset_folder_4test+'img4D_pairs.data'

if param.__train_on:
    triplet_stack_train = load_dataset_list(sampleIndx_list_train)
    print("before filterOutNearPurePatches_4D, triplet_stack_train.shape={}".format(triplet_stack_train.shape))
    triplet_stack_train = filterOutNearPurePatches_4D(triplet_stack_train)
    print("after filterOutNearPurePatches_4D, triplet_stack_train.shape={}".format(triplet_stack_train.shape))
if param.__val_on:
    triplet_stack_val = load_dataset_list(sampleIndx_list_val)
    print("before filterOutNearPurePatches_4D, triplet_stack_val.shape={}".format(triplet_stack_val.shape))
    triplet_stack_val = filterOutNearPurePatches_4D(triplet_stack_val)
    print("after filterOutNearPurePatches_4D, triplet_stack_val.shape={}".format(triplet_stack_val.shape))
if param.__test_on:
    img_stack_test = np.load(open(img_stack_test_file))        
    print('data is loaded: {}'.format(img_stack_test_file))


train_val_mean_acc = []

list_triplet_alpha = [0.2] if param.__debugON else [0.2] #[0.2,2,20,100]
list_DEFAULT_LR = [0.01, 0.001] if param.__debugON else [0.001]# 0.01
list_weight_decay = [0.003] if param.__debugON else [0.003] #[0.0003,0.003,0.3]

for list_param in list(itertools.product(list_triplet_alpha,list_DEFAULT_LR,list_weight_decay)): 
    param.__triplet_alpha = list_param[0]
    param.__DEFAULT_LR = list_param[1]
    param.__weight_decay = list_param[2]
    try:    # in case there are some errors, e.g. nan ... 
        main()
    except Exception: 
        print("exception"+str(sys.exc_info()))
        pass

    print("untile now, the train_val_mean_acc: {}".format(train_val_mean_acc))
