import numpy as np

# "train_fusionNet" "train_volumeNet_only" "test_fusionNet"  "reconstruct_model"
whatUWant = "reconstruct_model"
debug_ON = False
__define_fns = True if debug_ON else True

print "\ncurrent mode *** {} / debug_ON {} ***\n".format(whatUWant, debug_ON)


###########################
#   params rarely change
###########################
#------------ 
## params only for similarNet
#------------
__RETURN_ALL_VIEWPAIRS = 0 # <= 0

#------------ 
## common params 
#------------
# only do the preprocessing in the method 'preprocess_augment', don't subtract the mean in the network defination
__non_trainVal_set = [23,24,27,29,73,  114,118,  \
        25,26,27,  1,11,24,34,49,62,  11,32,33,48,75,\
        110,25,1,4,77,  1,9,10,12,13,15,33,54,  78,79,80,81] 
__trainVal_set = [i for i in range(1,129) if i not in __non_trainVal_set]
__rand_val_set = [35, 37, 43, 5, 66, 117, 17, 106, 21, 40, 82, 56, 86, 3, 67, 28, 38, 59] # 18 randomly selected models
__rand_train_set = [i for i in __trainVal_set if i not in __rand_val_set]
__CHANNEL_MEAN = np.asarray([123.68,  116.779,  103.939, 123.68,  116.779,  103.939]).astype(np.float32) # RGBRGB (VGG mean) [123.68,  116.779,  103.939, 123.68,  116.779,  103.939] [0]
__use_newLayerAPI_dilatConv = True
__read_partial_of_train_set = not debug_ON
__N_colorChannel = 3 # 3: rgb; 1: grey
__soft_gt = False
__soft_gt_thresh = 0.2
__only_nonOcclud_cubes = False
__surfPredict_scale4visual = 2
__root_path = '/home/mengqi/dataset/MVS/'
__visualizer = '/home/mengqi/working/program/4.1/20160725b-build/pcl_visualizer_demo'

__voxelVolume_txt_fld = __root_path + 'lasagne/samplesVoxelVolume/pcl_txt_50x50x50_2D_2_3D-without-offsurf/' # pcl_txt_50x50x50_2D_2_3D-emptyCube / pcl_txt_50x50x50_2D_2_3D-without-offsurf

__train_fusionNet = whatUWant is "train_fusionNet"

###########################
#   several modes:
#   "train_fusionNet" "train_volumeNet_only" "test_fusionNet"
###########################

if whatUWant is "train_fusionNet":
    # the main function includes these 4 parts
    __train_ON = True
    __val_ON = True
    __test_ON = False
    __reconstr_ON = False

    #------------ 
    ## params only for train / val
    #------------
    # only update the params in the range (l1,l2]. DON'T update l1's param
    # To update all params. Set to None
    __layer_range_tuple_2_update = ("feature_input", "output_softmaxWeights")
    # control whether the net_fn's inputlist includes weight_tensor
    # for each sample, we only consider this much random views
    __N_randViews4train = 4     # 4
    __N_randViews4val = __N_randViews4train
    # randomly select N view pairs from all the pair combinations
    __N_select_viewPairs2train = 6 #1/2 
    __N_select_viewPairs2val = __N_select_viewPairs2train
    # initial default learning rate
    __lr = 5
    # every N epoch, lr *= lr_decay 
    __lr_decay_per_N_epoch = 100
    __lr_decay = np.array([0.1]).astype(np.float32)
    # batch size 
    __chunk_len_train = 6#8
    __chunk_len_val = 5 if debug_ON else 12
    __num_epochs = 1000
    __train_set = [5,27,57,65,110,123] if debug_ON else __rand_train_set
    __val_set = [17] if debug_ON else __rand_val_set
    __val_visualize_ON = False
    __every_N_epoch_2saveModel = 1
    __layer_2_save_model = "output_fusionNet"

    #------------ 
    ## params only for similarNet
    #------------
    # each patch pair --> features to learn to decide view pairs
    # 2 * 128D/image patch + 1 * (dis)similarity + 1 * angle<v1,v2>
    __similNet_features_dim = 128*2+1+1
    __similNet_hidden_dim = 100
    # __pretrained_similNet_model_file = __root_path + 'lasagne/save_model_2views_triplet/epoch7_acc_tr0.795_val0.789.model' #{244678}
    # __pretrained_similNet_model_file = __root_path + 'lasagne/save_model_2views_triplet/epoch33_acc_tr0.707_val0.791.model' # allDTU
    __pretrained_similNet_model_file = __root_path + 'lasagne/save_model_2views_triplet/epoch49_acc_tr0.829_val0.842.model' # faceNet

    #------------ 
    ## common params 
    #------------
    # view index of the considered views
    __view_set = range(1,50) ## 1-49
    __use_pretrained_model = True
    if __use_pretrained_model:
        __layer_2_load_model = "output_volumeNet" # ["output_volumeNet_reshape","output_softmaxWeights"]#"output_volumeNet" ##output_fusionNet/fuse_op_reshape
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-62-0.588_0.95.model" #{8751de6} 3.2-origNet-newTrainSet
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-48-0.547_0.949.model" #{ca2962} new APIs with -=mean
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-82-0.629_0.949.model" # new APIs with -=mean, continuely trained on {ca2962}
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-3-0.856_0.949.model" # only trained end layers of fusionNet
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-114-0.606_0.95.model" # dafaa62, subtract VGG mean
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-22-0.781_0.958.model" # 94ba83, with off surf cubes (finetuned the dafaa62)
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-52-0.752_0.959.model" # a23e95e, with off surf cubes, new
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-18-0.555_0.953.model" # without off surface
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-4-4.48_0.948.model" # new allDTU val_acc = 3.73136
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-32-4.37_0.952.model" # new allDTU val_acc = 3.56935
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-32-4.17_0.952.model" # new allDTU val_acc = 3.3805
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-56-4.65_0.953.model" # new allDTU val_acc = 3.58896
        __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-52-0.752_0.959.model" # iccv

elif whatUWant is "train_volumeNet_only":
    # the main function includes these 4 parts
    __train_ON = True
    __val_ON = True
    __test_ON = False
    __reconstr_ON = False

    #------------ 
    ## params only for train / val
    #------------
    # only update the params in the range (l1,l2]. DON'T update l1's param
    # To update all params. Set to None
    __layer_range_tuple_2_update = None
    # control whether the net_fn's inputlist includes weight_tensor
    __train_fusionNet = False
    __view_set = range(1,50) ## 1-49
    # for each sample, we only consider this much random views
    __N_randViews4train = len(__view_set)-1 #4 / len(__view_set)-1
    __N_randViews4val = __N_randViews4train
    # randomly select N view pairs from all the pair combinations
    __N_select_viewPairs2train = 6 #1/2/6 # NO of view pair combinations
    __N_select_viewPairs2val = __N_select_viewPairs2train
    # initial default learning rate
    __lr = 5
    # every N epoch, lr *= lr_decay 
    __lr_decay_per_N_epoch = 100
    __lr_decay = np.array([0.1]).astype(np.float32)
    # batch size 
    __chunk_len_train = 6#8
    __chunk_len_val = 5 if debug_ON else 16
    __num_epochs = 1000
    __train_set = [5] if debug_ON else __rand_train_set  # [5,27,57,65,110,123]
    __val_set = [17] if debug_ON else __rand_val_set
    __val_visualize_ON = False
    __every_N_epoch_2saveModel = 100 if debug_ON else 2
    __layer_2_save_model = "output_volumeNet_channelPool" 

    #------------ 
    ## common params 
    #------------
    # view index of the considered views
    __view_set = range(1,50) ## 1-49
    __use_pretrained_model = False
    if __use_pretrained_model:
        __layer_2_load_model = "output_volumeNet" ##output_fusionNet/fuse_op_reshape
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-114-0.606_0.95.model" #{dafaa6} only for comparison
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-106-0.648_0.95.model" #{94db15}
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-96-0.574_0.953.model" #{94db15}
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-2-0.942_0.976.model" # allDTU 
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-18-0.555_0.953.model" # without off surface
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-186-0.717_0.94.model" # 
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-198-0.774_0.95.model" #{94db15}
        __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-74-0.801_0.952.model" # without-off-surf, val_acc: 79.4%

elif whatUWant is "test_fusionNet": 
    """
    In this mode, we'd like to visualize the fusionNet output.
    Using the pcl visualizer display pcds: gt/prediction/colored_gt/colored_cubes
    """
    # the main function includes these 4 parts
    __train_ON = False
    __val_ON = False
    __test_ON = True
    __reconstr_ON = False

    #------------ 
    ## params only for test
    #------------
    __chunk_len_test = 5
    # NO. of view pair 
    __N_select_viewPairs2test = 2 #1/2  
    __test_set = [32,33] # if debug_ON else [17] # [5] is used to check whether overfitting to learn smth
    # because of poor data loading function 'load_train_val_gt_asnp'
    __val_set = __test_set 
    __test_visualize_ON = True

    #------------ 
    ## params only for similarNet
    #------------
    # each patch pair --> features to learn to decide view pairs
    # 2 * 128D/image patch + 1 * (dis)similarity + 1 * angle<v1,v2>
    __similNet_features_dim = 128*2+1+1
    __similNet_hidden_dim = 100
    __pretrained_similNet_model_file = __root_path + 'lasagne/save_model_2views_triplet/epoch7_acc_tr0.795_val0.789.model' #{244678}

    #------------ 
    ## common params 
    #------------
    # view index of the considered views
    __view_set = range(1,50) ## 1-49
    __use_pretrained_model = True
    if __use_pretrained_model:
        __layer_2_load_model = ["output_volumeNet_reshape","output_softmaxWeights"] ##output_fusionNet/fuse_op_reshape
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-2-0.906_0.95.model" # {a8a67b} 3.2-fusionNet-updatesWeight
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-1-0.88_0.936.model" # debug
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-3-0.856_0.949.model" # 
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-9-0.904_0.949.model" # subtract VGG mean --> volumeNet --> only update fusionNet end layers 
        __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-7-0.956_0.96.model" # allDTU 

elif whatUWant is "reconstruct_model": 
    """
    In this mode, we'd like to reconstruct models using the similNet and fusionNet.
    """
    # the main function includes these 4 parts
    __train_ON = False
    __val_ON = False
    __test_ON = False
    __reconstr_ON = True

    #------------ 
    ## params only for reconstruction
    #------------
    __weighted_fusion = True # True: weighted average in the fusion layer; False: average
    __random_viewSelect = False
    if __weighted_fusion and __random_viewSelect: print("\nERROR:!!!!!!! __weighted_fusion and __random_viewSelect = True !!!!!!!!\n")
    __D_randcrop = 64 #32/64 # NO of voxels resolution along each axis
    __chunk_len_reconstr = {32:15, 64:2}[__D_randcrop] #15
    __D_center = {32:26, 64:52}[__D_randcrop] #26 ##20, 26 may speed up the process
    # NO. of view pair 
    __N_viewPairs = [5] # range(1,8,2) #6 #1/2  
    __test_set = [118]#[17]
    # because of poor data loading function 'load_train_val_gt_asnp'
    __val_set = __test_set 
    # __reconstr_model = __test_set[0]
    __weighted_fusion = False if __random_viewSelect else __weighted_fusion # if randomly select views, make sure to fuse using same weights.
    __patch_hw_size = 64
    __surfPredict_scale4reconstr = 2

    __planeMask_fld = __root_path + 'SampleSet/MVS Data/ObsMask/'

    # view index of the considered views
    __view_set = range(50,65) # range(50,65) # range(1,50) ## 1-49

    ## __grids_d = 50 # NO of voxels resolution in each axis
    ## __D_randcrop = 16 #32 ##5*D/6

    __min_prob = 1/2.1 # in order to save memory, filter out the voxels with prob < min_prob
    # cube_stride is proportional to resol* cube_stride_over_Dsize
    __resol = np.float32(0.4) #0.4 resolution / the distance between adjacent voxels
    __cube_stride_over_Dsize = 1/2. ## cube_stride / cube_Dsize  1./.5 

    __pcd_folder = __root_path + 'lasagne/iccv_cameraReady-reconstruction_result/weighted_{}-randView_{}-{}views-cubeSize{}/'.format(\
            'ON' if __weighted_fusion else 'OFF', 'ON' if __random_viewSelect else 'OFF', len(__view_set), __D_randcrop)
    
    # [(-300,300), (-300,300), (500,900)] ==> 201 / 144 banches ??? Messy enough
    # [(-200,220), (-250,220), (500,900)] ==> 114 banches
    # [(-200,200), (-250,200), (450, 800)] ==> 91 banches
    # with table segmentation:
    # [(-300,300), (-300,300), (400,1000)]
    __selfDefined_boundingBox = debug_ON
    __reconstr_sceneRange = [(0, 60), (-150, -100), ( 580,630)]

    __save_ply = debug_ON # don't save ply file when debug=False
    #------------ 
    ## params only for similarNet
    #------------
    # each patch pair --> features to learn to decide view pairs
    # 2 * 128D/image patch + 1 * (dis)similarity + 1 * angle<v1,v2>
    __similNet_features_dim = 128*2+1+1
    __similNet_hidden_dim = 100
    # __pretrained_similNet_model_file = __root_path + 'lasagne/save_model_2views_triplet/epoch7_acc_tr0.795_val0.789.model' #{244678}
    # __pretrained_similNet_model_file = __root_path + 'lasagne/save_model_2views_triplet/epoch33_acc_tr0.707_val0.791.model' # allDTU
    __pretrained_similNet_model_file = __root_path + 'lasagne/save_model_2views_triplet/epoch49_acc_tr0.829_val0.842.model' # faceNet

    #------------ 
    ## common params 
    #------------
    __use_pretrained_model = True
    if __use_pretrained_model:
        __layer_2_load_model = ["output_volumeNet_reshape","output_softmaxWeights"] ##output_fusionNet/fuse_op_reshape
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-2-0.906_0.95.model" # {a8a67b} 3.2-fusionNet-updatesWeight
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-1-0.88_0.936.model" # debug
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-9-0.904_0.949.model" # subtract VGG mean --> volumeNet --> only update fusionNet end layers 
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-2-0.932_0.957.model" # off surf
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-5-0.932_0.958.model" # off surf, epoch 5
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-10-0.923_0.957.model" # new off surf
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-7-0.956_0.96.model" # allDTU
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-3-0.915_0.95.model" # allDTU
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-9-0.921_0.953.model" # allDTU
        #__pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-19-0.918_0.951.model" # allDTU
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-4-5.43_0.95.model" # allDTU
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-3-7.21_5.33.model" # allDTU
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-2-7.23_5.26.model" 
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-3-6.84_5.47.model" 
        # __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-2-7.96_6.32.model"
        __pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-3-6.97_5.83.model"




__N_view = len(__view_set)

if '__train_set' not in dir():
    __train_set = [] 
if '__val_set' not in dir():
    __val_set = [] 
if '__test_set' not in dir():
    __test_set = [] 
if '__num_epochs' not in dir():
    __num_epochs = 2
if '__similNet_features_dim' not in dir():
    __similNet_features_dim = 66
if '__similNet_hidden_dim' not in dir():
    __similNet_hidden_dim = 666
if '__D_randcrop' not in dir():
    __D_randcrop = 32

__input_hwd = __D_randcrop

__pts_in_voxel_MAX = 10. # so use np.uint8 to save the train/val data file
__pts_in_voxel_MIN = 0.
__grid_D_4train = 50 # NO of voxels resolution along each axis


__cube_param_N = 5 # x,y,z,resol,modelIndx






__model_folder = __root_path + 'lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/'
#pretrained_model_file = model_folder+'modelfiles/2D_2_3D-2-0.767_0.947.model'
# pretrained_model_file = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-62-0.588_0.95.model" #{8751de6} 3.2-origNet-newTrainSet
pretrained_model_file_finetune = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-2-0.906_0.95.model" # 
# pretrained_model_file_only4ValTest = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-2-0.906_0.95.model" # {a8a67b} 3.2-fusionNet-updatesWeight
# pretrained_model_file_only4ValTest = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-12-0.906_0.949.model" # {c3536d} 3.2-fusionNet-moreModels-lr_coef-0.1_1
pretrained_model_file_only4ValTest = __root_path + "lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/2D_2_3D-12-0.895_0.949.model"  # {8aa907} 3.2-fusionNet-moreModels-lr_coef-1_1, 100 hidden neuron
pretrained_model_file = pretrained_model_file_finetune if __train_ON else pretrained_model_file_only4ValTest


#------------## only for debug       
__visualize_when_generateData_ON = False

__tempResult_4visual_fld = __root_path + 'lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/temp_visualization/'
__data_fld = __root_path + 'lasagne/samplesVoxelVolume/modelfile_50x50x50_2D_2_3D/'
__camera_po_txt_fld = __root_path + 'pos/'
__model_imgs_fld = __root_path + 'Rectified_mean/' #'{}Rectified_mean/'.format('backup/' if whatUWant is "train_fusionNet" else '' ) #backup/Rectified_mean
#__model_imgs_npz_fld = __root_path + 'Rectified_mean/npz/'

__pcl_pcd2ply_exe = '~/Downloads/pcl-trunk/build/bin/pcl_pcd2ply'




## print the params in log
for _var in dir():
    if '__' in _var and not (_var[-2:] == '__'): # don't show the uncustomed variables, like '__builtins__'
        exec("print '{} = '.format(_var), "+_var)





