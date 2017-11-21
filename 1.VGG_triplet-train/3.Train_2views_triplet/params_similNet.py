""" define const parameters """

import numpy as np

__train_on = True
__val_on = True
__test_on = False
__mini_dataset = False


__debugON = False

__shuffle_sample_ON = True
__save_weight_ON = True if not __debugON else False

__use_pretrained_model = True
__use_VGG16_weights = True

__only_train_few_layers = True
__batch_size_train = 200 if __only_train_few_layers else 300
__batch_size_val = 500 if __only_train_few_layers else 100      # why this is even smaller than that for training?
__batch_size_test = 1500 if __only_train_few_layers else 300

__featureDim = 128 # dim of the embedding
__triplet_alpha = 0.2
__DEFAULT_LR = 0 # will be updated during param tuning
__weight_decay = 0.0001

__hw_size = 64

__train_epoch = 20 if __debugON else 2000

#model = pickle.load(open('/home/mji/theano/lasagne/Recipes/examples/VGG16/vgg16.pkl'))
#MEAN_IMAGE_BGR = model['mean value'].astype(np.float32)
__MEAN_IMAGE_BGR = np.asarray([103.939,  116.779,  123.68]).astype(np.float32)

__featureDim = 128


__squared_L2_dist = True # in the FaceNet paper: squared L2 distance <---> similarity


