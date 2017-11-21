import sys
import os
sys.path.insert(0,'./Lasagne') # local checkout of Lasagne
import lasagne
import theano
import gzip
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from train_val_data import *
import random
import pickle
sys.path.append("../../1.VGG_triplet-train/3.Train_2views_triplet")
import similarityNet
sys.path.append("../../1.VGG_triplet-train/2.vgg_prepare_py")
import prepare_data
sys.path.append("../../4.2D_2_3D-test/3.adapt_thresh")
import sparseCubes
import pdb
import params_volume
import nets
import time
import thread
import utils
import scipy.io

random.seed(201605)
np.random.seed(201605)




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
    if not os.path.exists(params_volume.__model_folder):
        os.mkdir(params_volume.__model_folder)
    with open(params_volume.__model_folder+filename, 'wb') as f:
        pickle.dump(data, f)
        print("save model to: {}{}".format(params_volume.__model_folder, filename))

def add_noise(X):
    shape = X.shape
    X_flat = X.flatten()
    N_indx = np.random.randint(0, X.size, X.size/2) 
    N = np.random.normal(0,.5,X.size/2)
    X_flat[N_indx] = N
    
    Hole_indx = np.random.randint(0, X.size, X.size/5) 
    X_flat[Hole_indx] = 0
    return X_flat.reshape(shape)



def preprocess_augmentation(gt_sub, X_sub_rgb, augment_ON = True, crop_ON = True, color2grey = True):
    # X_sub /= 255.
    mean_color = params_volume.__CHANNEL_MEAN[None,:,None,None,None]
    _shape = X_sub_rgb.shape
    X_sub_rgb = X_sub_rgb.astype(np.float32)
    if color2grey:
        X_sub = np.tensordot(X_sub_rgb.reshape(_shape[:1]+(2,3)+_shape[2:]), 
                             np.array([0.299,0.587,0.114]).astype(np.float32), 
                             axes=([2],[0])) # convert RGB to grey (N,6,D,D,D)==> (N,2,3,D,D,D)==> (N,2,D,D,D)
    else:
        X_sub = np.copy(X_sub_rgb)
    X_sub -= mean_color  

    if augment_ON:
        X_sub += np.random.randint(-30,30,1) # illumination argmentation
        X_sub += np.random.randint(-5,5,X_sub.shape) # color noise
        gt_sub, X_sub = data_augment_rand_rotate(gt_sub, X_sub) # randly rotate multiple times
        gt_sub, X_sub = data_augment_rand_rotate(gt_sub, X_sub)
        gt_sub, X_sub = data_augment_rand_rotate(gt_sub, X_sub)
        ##gt_sub, X_sub = data_augment_scipy_rand_rotate(gt_sub, X_sub) ## take a lot of time
    if crop_ON:
        gt_sub, X_sub, X_sub_rgb= data_augment_crop([gt_sub, X_sub, X_sub_rgb], random_crop=augment_ON) # smaller size cube       

    return gt_sub, X_sub, X_sub_rgb

def save_pcd_pcd2ply(points2save, pcl_pcd2ply_exe, pcd_folder, filename ):
    print("saving ply")
    RECONSTR_save_pcd(points2save, pcd_folder, filename)
    os.system( "{} {} {}".format(pcl_pcd2ply_exe, pcd_folder+filename,\
            (pcd_folder+filename).replace('.pcd','.ply')))


def deepSurf():
        
    #X_train_noisy, X_train_gt, X_val_noisy, X_val_gt = load_3Ddata(LowResolution=False)
    cameraPOs = load_all_cameraPO_files_f64()
    cameraTs = prepare_data.cameraPs2Ts(cameraPOs)
    if params_volume.__val_ON or params_volume.__test_ON:
        val_gt_param, val_gt_01, val_gt_float, val_gt_visib = load_gt_asnp(params_volume.__val_set, visualization_ON=False, return_soft_occupancy_ON = True)
        test_gt_param, test_gt_01, test_gt_float, test_gt_visib = val_gt_param, val_gt_01, val_gt_float, val_gt_visib



    ##mean1D = X_train_noisy.flatten().mean()
    ##mean4D = X_train_noisy.mean(axis=0)
    ##mean5D = mean4D[None,:]
    mean = 0   # mean1D/5D
    mean_visualizer = 0    # mean1D/4D
    ##X_train_noisy -= mean
    ##X_train_gt -= mean
    ##X_val_noisy -= mean
    ##X_val_gt -= mean

#===============================================
    # define the nets and the fns

    if params_volume.__train_ON or params_volume.__val_ON:
        models_img_val = load_modellist_meanIMG(params_volume.__val_set)
        lr_tensor = theano.shared(np.array(params_volume.__lr, dtype=theano.config.floatX))         
        if params_volume.__define_fns:
            net, train_fn, val_fn = nets.def_net_fn_train_val(return_train_fn=params_volume.__train_ON, return_val_fn=params_volume.__val_ON, \
                    with_weight=params_volume.__train_fusionNet, \
                    input_cube_size = params_volume.__input_hwd, N_samples_perGroup = params_volume.__N_select_viewPairs2train, \
                    Dim_feature = params_volume.__similNet_features_dim, num_hidden_units = params_volume.__similNet_hidden_dim, \
                    default_lr = lr_tensor)
                

    if params_volume.__test_ON:
        models_img_test = load_modellist_meanIMG(params_volume.__test_set)
        net, fuseNet_calcWeight_fn, fuseNet_fn = nets.def_net_fn_test(with_weight=True, \
                input_cube_size = params_volume.__input_hwd, N_samples_perGroup = params_volume.__N_select_viewPairs2test, \
                Dim_feature = params_volume.__similNet_features_dim, num_hidden_units = params_volume.__similNet_hidden_dim)

    if params_volume.__reconstr_ON:
        models_img_reconstr = load_modellist_meanIMG(params_volume.__test_set)
        net, fuseNet_calcWeight_fn, fuseNet_fn = nets.def_net_fn_test(with_weight=True, with_groundTruth=False, \
                input_cube_size = params_volume.__input_hwd, N_samples_perGroup = N_viewPairs, \
                Dim_feature = params_volume.__similNet_features_dim, num_hidden_units = params_volume.__similNet_hidden_dim,\
                return_unfused_predict = True)

#===============================================
    # load the pretrained model
    if params_volume.__use_pretrained_model == True:
        model_file = params_volume.__pretrained_model_file
        print ('loading volumeNet / fusionNet model: {}'.format(model_file))
        if not isinstance(params_volume.__layer_2_load_model, list):
            layer_names_2_load_model = [params_volume.__layer_2_load_model]
        else:
            layer_names_2_load_model = params_volume.__layer_2_load_model
        layers_2_load_model = [net[_layer_name] for _layer_name in layer_names_2_load_model]
        load_entire_model(layers_2_load_model, model_file)      ##output_fusionNet/fuse_op_reshape

#===============================================  
    # define the similarity net

    if params_volume.__test_ON or params_volume.__reconstr_ON:
    # in the test mode, all the patchPair-feature of all the view pairs will be calculated
    # use this new version to speed up. This new version will slow down the train/val
    # because train/val only need specified few view pairs, don't need to calculate the patch features of all the views. 
        net_featureEmbedLayer,net_featurePair2simil_similarityLayer, patch2feature_fn,featurePair2simil_fn = \
                similarityNet.def_patch_TO_feature_TO_similarity_net_fn()

        with open(params_volume.__pretrained_similNet_model_file) as f:
            data = pickle.load(f)
            lasagne.layers.set_all_param_values([net_featureEmbedLayer,net_featurePair2simil_similarityLayer], data) #[similNet_outputLayer]
            print('loaded similNet model: {}'.format(params_volume.__pretrained_similNet_model_file))

    if params_volume.whatUWant == 'train_fusionNet' and (params_volume.__train_ON or params_volume.__val_ON):
        similNet_outputLayer, similNet_fn = similarityNet.def_patchPair_TO_feature_simil_net_fn()

        with open(params_volume.__pretrained_similNet_model_file) as f:
            data = pickle.load(f)
            lasagne.layers.set_all_param_values([similNet_outputLayer], data)
            print('loaded similNet model: {}'.format(params_volume.__pretrained_similNet_model_file))




#===============================================       
    
    if params_volume.__val_ON:
        val_gt = val_gt_float if params_volume.__soft_gt else val_gt_01 # val_gt_float/val_gt_01
        inScope_indx = inScope_Cubes(N_inScopeViews = params_volume.__N_randViews4val, occupiedCubes_param = val_gt_param, \
                                          cameraPOs=cameraPOs, models_img=models_img_val, cubes_visib = None if params_volume.__train_fusionNet else val_gt_visib)        #len(params_volume.__view_set)-1
        val_gt_param, val_gt, val_gt_visib = [i[inScope_indx] for i in [val_gt_param, val_gt, val_gt_visib]]
        N_val = val_gt.shape[0]
    if params_volume.__test_ON:
        test_gt = test_gt_float if params_volume.__soft_gt else test_gt_01 # test_gt_float/test_gt_01
        inScope_indx = inScope_Cubes(N_inScopeViews = len(params_volume.__view_set)-1, occupiedCubes_param = test_gt_param, \
                                           cameraPOs=cameraPOs, models_img=models_img_test, cubes_visib = None)        
        test_gt_param, test_gt, test_gt_visib = [i[inScope_indx] for i in [test_gt_param, test_gt, test_gt_visib]]
        N_test = test_gt.shape[0]
    if params_volume.__train_ON:
        if not params_volume.__read_partial_of_train_set:
            train_set = params_volume.__train_set
            train_gt_param, train_gt_01, train_gt_float, train_gt_visib = load_gt_asnp(train_set, visualization_ON=False, return_soft_occupancy_ON = True)
            train_gt = train_gt_float if params_volume.__soft_gt else train_gt_01 # train_gt_float/train_gt_01
            models_img_train = load_modellist_meanIMG(train_set)
            # when train volumeNet, set cubes_visib = gt_visib to make sure there are enough non-occluded views.
            # when train fusionNet, the fusionNet should learn how the occluded view pairs looks like, so only set cubes_visib = None.
            inScope_indx = inScope_Cubes(N_inScopeViews = params_volume.__N_randViews4train, occupiedCubes_param = train_gt_param, \
                                          cameraPOs=cameraPOs, models_img=models_img_train, cubes_visib = train_gt_visib if params_volume.__only_nonOcclud_cubes else None)    
            train_gt_param, train_gt, train_gt_visib = [i[inScope_indx] for i in [train_gt_param, train_gt, train_gt_visib]]
            N_train = train_gt.shape[0]

    input_is_grey = params_volume.__N_colorChannel == 1
    for epoch in range(1, params_volume.__num_epochs):
        
        if params_volume.__train_ON:
            print "starting training..." 
            # only load partial of the training models in each epoch because of the limited PC memory (or poor programming).
            if params_volume.__read_partial_of_train_set:
                train_set_partial = random.sample(params_volume.__train_set, max(1,len(params_volume.__train_set)/3))
                train_gt_param, train_gt_01, train_gt_float, train_gt_visib = load_gt_asnp(train_set_partial, visualization_ON=False, return_soft_occupancy_ON = True)
                train_gt = train_gt_float if params_volume.__soft_gt else train_gt_01 # train_gt_float/train_gt_01
                models_img_train = load_modellist_meanIMG(train_set_partial)
                inScope_indx = inScope_Cubes(N_inScopeViews = params_volume.__N_randViews4train, occupiedCubes_param = train_gt_param, \
                                           cameraPOs=cameraPOs, models_img=models_img_train, cubes_visib = train_gt_visib if params_volume.__only_nonOcclud_cubes else None)    
                
                train_gt_param, train_gt, train_gt_visib = [i[inScope_indx] for i in [train_gt_param, train_gt, train_gt_visib]]
                N_train = train_gt.shape[0]
            if epoch%params_volume.__lr_decay_per_N_epoch == 0:
                lr_tensor.set_value(lr_tensor.get_value() * params_volume.__lr_decay)        
                print 'current updated lr_tensor = {}'.format(lr_tensor.get_value())
                
            acc_train_batches = []
            acc_guess_all0 = []
            for batch in range(1, N_train/params_volume.__chunk_len_train): ##3 or N_train/params_volume.__chunk_len_train
                selected = random.sample(range(0,N_train),params_volume.__chunk_len_train) ##almost like shuffle
                ##selected = list(set(np.random.random_integers(0,N_train-1,params_volume.__chunk_len_train*2)))[:params_volume.__chunk_len_train] ## set([2,2,3])=[2,3], 
                train_gt_sub = train_gt[selected][:,None,...] ## convert to 5D
                ## do some simple data augmentation in each batch, save back to the data_array, which will be shuffled next time

                if params_volume.__train_fusionNet:
                    selected_viewPairs, similNet_features = perform_similNet(similNet_fn=similNet_fn, \
                            occupiedCubes_param = train_gt_param[selected], N_select_viewPairs = params_volume.__N_select_viewPairs2train, models_img=models_img_train, \
                            view_set = params_volume.__view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=32, batch_size=100, similNet_features_dim = params_volume.__similNet_features_dim)
                    
                    train_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = train_gt_param[selected], \
                            cameraPOs=cameraPOs, models_img=models_img_train, visualization_ON = False, occupiedCubes_01 = train_gt_sub)
                    train_gt_sub, train_X_sub, train_X_rgb_sub = preprocess_augmentation(train_gt_sub, train_X_sub, augment_ON=True, color2grey = input_is_grey)
                    _loss, acc, predict_train, similFeature_softmax_output = train_fn(train_X_sub, similNet_features.reshape(-1,similNet_features.shape[-1]), train_gt_sub)

                else:
                    selected_viewPairs = select_M_viewPairs_from_N_randViews_for_N_samples(params_volume.__view_set, params_volume.__N_randViews4train, \
                            params_volume.__N_select_viewPairs2train, N_samples = params_volume.__chunk_len_train, \
                            cubes_visib = train_gt_visib[selected] if params_volume.__only_nonOcclud_cubes else None)
                    # selected_viewPairs = np.random.choice(params_volume.__view_set, (params_volume.__chunk_len_train,params_volume.__N_select_viewPairs2train,2))
                    train_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = train_gt_param[selected], \
                            cameraPOs=cameraPOs, models_img=models_img_train, visualization_ON = False, occupiedCubes_01 = train_gt_sub)
                    train_gt_sub, train_X_sub, train_X_rgb_sub = preprocess_augmentation(train_gt_sub, train_X_sub, augment_ON=True, color2grey = input_is_grey)
                    _loss, acc, predict_train = train_fn(train_X_sub, train_gt_sub)
                    

                acc_train_batches.append(list(acc))
                acc_guess_all0.append(1-float(train_gt_sub.sum())/train_gt_sub.size)
                print("Epoch %d, batch %d: Loss %g, acc %g, acc_guess_all0 %g" % \
                                              (epoch, batch, np.sum(_loss), np.asarray(acc_train_batches).mean(), np.asarray(acc_guess_all0).mean()))
                
        
        if params_volume.__val_ON:
            if (epoch % 1) == 0:    # every N epoch
                print "starting validation..."    
                acc_val_batches = []

                for batch_val in range(0, N_val/params_volume.__chunk_len_val):
                    selected = range(batch_val*params_volume.__chunk_len_val,(batch_val+1)*params_volume.__chunk_len_val)
                    val_gt_sub = val_gt[selected][:,None,...] ## convert to 5D

                    if params_volume.__train_fusionNet:
                        selected_viewPairs, similNet_features = perform_similNet(similNet_fn=similNet_fn, \
                                occupiedCubes_param = val_gt_param[selected], N_select_viewPairs = params_volume.__N_select_viewPairs2val, models_img=models_img_val, \
                                view_set = params_volume.__view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=32, batch_size=100, similNet_features_dim = params_volume.__similNet_features_dim)
                        
                        val_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = val_gt_param[selected], \
                                cameraPOs=cameraPOs, models_img=models_img_val, visualization_ON = False, occupiedCubes_01 = val_gt_sub)                    
                        val_gt_sub, val_X_sub, val_X_rgb_sub = preprocess_augmentation(val_gt_sub, val_X_sub, augment_ON=False, color2grey = input_is_grey)
                        acc_val, predict_val = val_fn(val_X_sub, similNet_features.reshape(-1,similNet_features.shape[-1]), val_gt_sub)
                        # acc_val, predict_val = fuseNet_val_fn(val_X_sub, val_gt_sub)
                    else:
                        selected_viewPairs = select_M_viewPairs_from_N_randViews_for_N_samples(params_volume.__view_set, params_volume.__N_randViews4val, \
                                params_volume.__N_select_viewPairs2val, cubes_visib = val_gt_visib[selected] if params_volume.__only_nonOcclud_cubes else None)
                        # selected_viewPairs = np.random.choice(params_volume.__view_set, (params_volume.__chunk_len_val,params_volume.__N_select_viewPairs2val,2))
                        val_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = val_gt_param[selected], \
                                cameraPOs=cameraPOs, models_img=models_img_val, visualization_ON = False, occupiedCubes_01 = val_gt_sub)                    
                        val_gt_sub, val_X_sub, val_X_rgb_sub = preprocess_augmentation(val_gt_sub, val_X_sub, augment_ON=False, color2grey = input_is_grey)
                        acc_val, predict_val = val_fn(val_X_sub, val_gt_sub)
                        
                        
                    acc_val_batches.append(acc_val)   
                    if params_volume.__val_visualize_ON :
                        X_1, X_2 = [val_X_rgb_sub[0:params_volume.__N_select_viewPairs2val], val_gt_sub[0:1]]
                        result = predict_val[0]
                        X_1 += 0 #params_volume.__CHANNEL_MEAN # [None,:,None,None,None]   #(X_1+.5)*255. 
                        tmp_5D = np.copy(X_1) # used for visualize the surface part of the colored cubes
                        # if want to visualize the result, just stop at the following code, and in the debug probe run the visualize_N_densities_pcl func.
                        # rimember to 'enter' before continue the program~
                        tmp_5D[:,:,X_2[0].squeeze()==0]=0 
                        if not params_volume.__train_ON:
                            visualize_N_densities_pcl([X_2[0]*params_volume.__surfPredict_scale4visual, result*params_volume.__surfPredict_scale4visual, tmp_5D[0,3:], tmp_5D[0,:3], X_1[0,3:], X_1[0,:3]])
                acc_val = np.asarray(acc_val_batches).mean()
                print("val_acc %g" %(acc_val))

            if (epoch % params_volume.__every_N_epoch_2saveModel) == 0:
                save_entire_model(net[params_volume.__layer_2_save_model], '2D_2_3D-{}-{:0.3}_{:0.3}.model'.format(epoch, np.asarray(acc_train_batches).mean(), acc_val))             

        if params_volume.__test_ON:       
            print "starting testing..."    
            acc_test_batches = []

            for batch_test in range(0, N_test/params_volume.__chunk_len_test):
                selected = range(batch_test*params_volume.__chunk_len_test,(batch_test+1)*params_volume.__chunk_len_test)
                test_gt_sub = test_gt[selected][:,None,...] ## convert to 5D
                start_time = time.time()
                all_viewPairs, similNet_features = perform_similNet(patch2feature_fn=patch2feature_fn,featurePair2simil_fn=featurePair2simil_fn, \
                        occupiedCubes_param = test_gt_param[selected], N_select_viewPairs = params_volume.__RETURN_ALL_VIEWPAIRS, models_img=models_img_test, \
                        view_set = params_volume.__view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=32, batch_size=100, similNet_features_dim = params_volume.__similNet_features_dim)
                print("perform_similNet takes {}".format(time.time() - start_time))
                all_similNet_weight = fuseNet_calcWeight_fn(similNet_features.reshape(-1,similNet_features.shape[-1]), \
                        n_samples_perGroup = all_viewPairs.shape[1]) # the result will have shape (N_pts, N_randPairs) 

                selected_viewPairs, selected_similNet_weight = select_N_argmax_viewPairs(all_viewPairs, all_similNet_weight, N=params_volume.__N_select_viewPairs2test)
                ## test_X_sub = gen_coloredCubes(N_randViews = N_randViews, N_viewPairs = N_select_viewPairs, occupiedCubes_param = test_gt_param[selected], \
                test_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = test_gt_param[selected], \
                        cameraPOs=cameraPOs, models_img=models_img_test, visualization_ON = False, occupiedCubes_01 = test_gt_sub)                    

                test_gt_sub, test_X_sub, test_X_rgb_sub = preprocess_augmentation(test_gt_sub, test_X_sub, augment_ON=False, color2grey = input_is_grey)
                # acc_test, predict_test = fuseNet_test_fn(test_X_sub, similNet_features.reshape(-1,similNet_features.shape[-1]), test_gt_sub)
                if selected_similNet_weight.shape[1] == 1:
                    acc_test, predict_test = fuseNet_fn(test_X_sub, test_gt_sub)
                else:
                    acc_test, predict_test = fuseNet_fn(test_X_sub, selected_similNet_weight, test_gt_sub)

                # acc_test, predict_test = fuseNet_test_fn(test_X_sub, test_gt_sub)
                acc_test_batches.append(list(acc_test))
                if params_volume.__test_visualize_ON :
                    X_1, X_2 = [test_X_rgb_sub[0:params_volume.__N_select_viewPairs2test], test_gt_sub[0:1]]
                    result = predict_test[0]
                    X_1 += 0 #params_volume.__CHANNEL_MEAN #[None,:,None,None,None]
                    tmp_5D = np.copy(X_1) # used for visualize the surface part of the colored cubes
                    # if want to visualize the result, just stop at the following code, and in the debug probe run the visualize_N_densities_pcl func.
                    # rimember to 'enter' before continue the program~
                    tmp_5D[:,:,X_2[0].squeeze()==0]=0 
                    if not params_volume.__train_ON:
                        visualize_N_densities_pcl([X_2[0]*params_volume.__surfPredict_scale4visual, result*params_volume.__surfPredict_scale4visual, tmp_5D[0,:3], tmp_5D[0,3:], X_1[0,:3], X_1[0,3:]])
            print("test_acc %g" %(np.asarray(acc_test_batches).mean()))

#=============================================== 
        if params_volume.__reconstr_ON:  
            print "starting reconstruction..."  
            sys.path.append("../../4.2D_2_3D-test/2.py-reconstruction/")
            import train_val_data_4Test # latter should merge this to 3.2   
            pcd_folder = params_volume.__pcd_folder
            save_folder = pcd_folder + "results_cubes_{}BB/".format(\
                    'self' if params_volume.__selfDefined_boundingBox else 'model')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)          
            start_time = time.time()
            points = [] 
            saved_params = []
            saved_prediction = []
            saved_maskID = []
            saved_selected_pairIndx = []
            saved_rgb = []
            prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list = [], [], [], []
            cube_ijk_np, param_np, viewPair_np = None, None, None

            # bounding box of the scene !
            # 
            BB_file_path = params_volume.__planeMask_fld + 'ObsMask{}_10.mat'.format(reconstr_modelIndx)
            BB_file = scipy.io.loadmat(BB_file_path)
            BB = BB_file['BB'] # (2,3)
            BB[0,:] -= 50
            BB[1,:] += 50 # little bit larger than the BB.
            
            Samples_param_np, cube_Dsize = RECONSTR_generate_3DSamples_param_np(reconstr_modelIndx,\
                    resol=params_volume.__resol, D_randcrop=params_volume.__D_randcrop, D_center=params_volume.__D_center,\
                    cube_stride_over_Dsize=params_volume.__cube_stride_over_Dsize, BB = None if params_volume.__selfDefined_boundingBox else BB)
            ##Nx,Ny,Nz = Samples_param_np[5].shape ## Nx * Ny * Nz = # of 3D cubes
            #### store the ix/iy/iz th cube's points in points_3Dlist[ix][iy][iz]. 
            ##refine_cubes_points_3Dlist = [[[[] for z in xrange(Nz)] for y in xrange(Ny)] for x in xrange(Nx)] 
            ##refine_cubes_occupancy = np.zeros((Nx,Ny,Nz)).astype(np.bool)
            ##refine_cubes_NOofParams = 3   ## adaptive threshld; # of nonempty neighbors; # of connected cubes
            ##refine_params = np.zeros((Nx,Ny,Nz,refine_cubes_NOofParams))
            
            # table segmentation
            # if the table segment file
            planeMask_file_path = params_volume.__planeMask_fld + 'Plane{}.mat'.format(reconstr_modelIndx)
            try:
                planeMask = scipy.io.loadmat(planeMask_file_path)
                planeMask_P = planeMask['P'][:,0]# (4,1) --> (4,)
                onPlane_selector = np.dot(np.c_[Samples_param_np[:,:3], np.ones(Samples_param_np.shape[0])], planeMask_P) > 0
                Samples_param_np = Samples_param_np[onPlane_selector]
            except:
                print("Warning: this model don't have plane mask to segment the table.'")
            #np.random.shuffle(Samples_param_np)
            print "starting VGG_triplet Net to predict the feature vectors for each view patches."
            pt_batch_N = 1000 if params_volume.debug_ON else 5000
            _iterations = Samples_param_np.shape[0] / pt_batch_N
            for _i, Samples_select in enumerate(utils.gen_batch_index(Samples_param_np.shape[0], pt_batch_N)):
                _param = Samples_param_np[Samples_select]
                # TODO: this part and other parts, like generate_coloredCubes, perform_similNet, and train_val_data_4Test.RECONSTR_select_valid_pts_in_cube use xyz as xyz_min
                # with += cube_Dsize/2
                _, desicion_onSurf = train_val_data_4Test.get_VGG_triplet_Net_featureVec(modelIndx=reconstr_modelIndx,\
                        view_set=params_volume.__view_set, hw_size = params_volume.__patch_hw_size,\
                        xyz_np = _param[:,:3] + cube_Dsize/2., patch2feature_fn=patch2feature_fn,featurePair2simil_fn=featurePair2simil_fn) 
                        #xyz_nplist=[i + D_center*_param[:,3]/2 for i in [_param[:,0],_param[:,1],_param[:,2]]])
                # TODO: inScope_nearSurf_Cubes may not neccessary
                ## test_param, inScope_indices = train_val_data_4Test.inScope_nearSurf_Cubes(N_inScopeViews = 2, occupiedCubes_param = _param, \
                ##                            VGGOccupancy=desicion_onSurf, cameraPOs=cameraPOs, models_img=models_img)           
                if desicion_onSurf is None:
                    continue
                test_param = _param[desicion_onSurf]
                N_test = test_param.shape[0]

                print '\nThere are {} inScope_nearSurf_Cubes will be processed. {} / {}'.format(N_test, _i, _iterations)

                chunk_len_test = params_volume.__chunk_len_reconstr
                for selected in utils.gen_batch_index(N_test, chunk_len_test):
                    ###TODO: currenly the similNet calculate the feature twice, here is the second time. First time is in the coarse-pcd generation prcessing
                    all_viewPairs, similNet_features = perform_similNet(patch2feature_fn=patch2feature_fn,featurePair2simil_fn=featurePair2simil_fn, \
                            occupiedCubes_param = test_param[selected], N_select_viewPairs = params_volume.__RETURN_ALL_VIEWPAIRS, models_img=models_img_reconstr, \
                            view_set = params_volume.__view_set, cameraPOs=cameraPOs, cameraTs=cameraTs, patch_r=32, batch_size=100, similNet_features_dim = params_volume.__similNet_features_dim)

                    if params_volume.__random_viewSelect:
                        all_similNet_weight = np.random.random(similNet_features.shape[:2]).astype(np.float32)
                    else:
                        all_similNet_weight = fuseNet_calcWeight_fn(similNet_features.reshape(-1,similNet_features.shape[-1]), \
                                n_samples_perGroup = all_viewPairs.shape[1]) # the result will have shape (N_pts, N_randPairs) 
                    selected_viewPairs, selected_similNet_weight = select_N_argmax_viewPairs(all_viewPairs, all_similNet_weight, N=N_viewPairs)
                    if not params_volume.__weighted_fusion:
                        selected_similNet_weight[:] = 1.0 / N_viewPairs
                    ###TODO: in the test process, the generated coloredCubes could be the exact size we want. Don't need to crop in the preprocess method. 
                    test_X_sub = gen_coloredCubes(selected_viewPairs = selected_viewPairs, occupiedCubes_param = test_param[selected], colorize_cube_D = params_volume.__D_randcrop,\
                                    cameraPOs=cameraPOs, models_img=models_img_reconstr, visualization_ON = False, return_pixIDmask_cubes_ON=False)
                    _, test_X_sub, test_X_rgb_sub = preprocess_augmentation(None, test_X_sub, augment_ON=False, crop_ON = False, color2grey = input_is_grey)
                    # TODO: this prediction process has lot redundant computation: if there are only few view pairs are valid, many view pairs will be calculated and latter filtered by 0/1_weight.
                    # a good solution is that: if there are few view pairs, only let these valid views go through the network.
                    predict_test, unfused_predictions = fuseNet_fn(test_X_sub) if N_viewPairs == 1 \
                            else fuseNet_fn(test_X_sub,selected_similNet_weight)

                    # rgb
                    # test_X_sub_highest_weight = test_X_sub.reshape((-1,N_viewPairs)+test_X_sub.shape[1:])[:,-1,...] # (N_pt*N_pairs, 6, D,D,D) ==> (N_pt,N_pairs, 6, D,D,D)[:,-1,...] which has the largest weight
                    # test_X_sub_highest_weight += params_volume.__CHANNEL_MEAN[None,:,None,None,None]
                    # rgb_test = (test_X_sub_highest_weight[:,:3] + test_X_sub_highest_weight[:,3:])/2 # RGB
                    test_X_rgb_sub += 0 #params_volume.__CHANNEL_MEAN #[None,:,None,None,None]
                    N_cubes = predict_test.shape[0]
                    # (N_cubes * N_viewPairs, 6, D,D,D), (N_cubes, N_viewPairs, D,D,D), (N_cubes, N_viewPairs) ==> (N_cubes, 3, D,D,D)
                    rgb_test = utils.generate_voxelLevelWeighted_coloredCubes(viewPair_coloredCubes = test_X_rgb_sub, \
                            viewPair_surf_predictions = unfused_predictions, weight4viewPair = selected_similNet_weight)

                    time_appendSparse = time.time()
                    updated_sparse_list_np = sparseCubes.append_dense_2sparseList(prediction_sub = predict_test, rgb_sub = rgb_test, param_sub = test_param[selected],\
                                    viewPair_sub = selected_viewPairs, min_prob = params_volume.__min_prob, rayPool_thresh = 0,\
                                    enable_centerCrop = True, D_center = params_volume.__D_center,\
                                    enable_rayPooling = True, cameraPOs = cameraPOs, cameraTs = cameraTs, \
                                    prediction_list = prediction_list, rgb_list = rgb_list, vxl_ijk_list = vxl_ijk_list, \
                                    rayPooling_votes_list = rayPooling_votes_list, \
                                    cube_ijk_np = cube_ijk_np, param_np = param_np, viewPair_np = viewPair_np)
                    prediction_list, rgb_list, vxl_ijk_list, rayPooling_votes_list, \
                            cube_ijk_np, param_np, viewPair_np = updated_sparse_list_np
                    # print("appending sparse results takes {:.3f}s".format(time.time() - time_appendSparse))


                print("\n this batch took {:.3f}s\n".format(time.time() - start_time))

            if params_volume.__save_ply:
                time_ply = time.time()
                ply_filename = os.path.join(save_folder,'model{}-{}viewPairs.ply'.format(reconstr_modelIndx, \
                        N_viewPairs))
                vxl_leftIndx_list = sparseCubes.filter_voxels(vxl_leftIndx_list=[],prediction_list=None, prob_thresh=None,\
                        rayPooling_votes_list=rayPooling_votes_list, rayPool_thresh=N_viewPairs)
                sparseCubes.save_sparseCubes_2ply(vxl_leftIndx_list, vxl_ijk_list, rgb_list, \
                        param_np, ply_filePath=ply_filename, normal_list=None)
                print("save ply takes {:.3f}s".format(time.time() - time_ply))

            time_npz = time.time()
            save_npz_file_path = os.path.join(save_folder,'model{}-{}viewPairs.npz'.format(reconstr_modelIndx, \
                        N_viewPairs))
            sparseCubes.save_sparseCubes(save_npz_file_path, *updated_sparse_list_np)
            print("save npz takes {:.3f}s".format(time.time() - time_npz))
                

        if not params_volume.__train_ON: ## only run one epoch
            break
        
if __name__ == "__main__":
    if params_volume.whatUWant == 'reconstruct_model':
        for _model in params_volume.__test_set:
            for _n in params_volume.__N_viewPairs:
                print('\n ********* \n model {}, N_viewPairs = {} \n ********* \n'.format(_model, _n))
                N_viewPairs = _n
                reconstr_modelIndx = _model
                deepSurf()
    else:
        deepSurf()




