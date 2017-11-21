
import lasagne
import theano
import theano.tensor as T
import numpy as np
np.random.seed(201601)


class Norm_L2_Layer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        input_sqr = input**2
        input_L2 = (input_sqr.sum(axis=1))**.5
        input_unit = input/input_L2[:,None]        
        return input_unit

def get_dist_from_feature_embed(feature_embed):
    diff_pos = feature_embed[::3,] - feature_embed[1::3,]
    diff_neg = feature_embed[::3,] - feature_embed[2::3,]
    diff_pos_sq_sum = (diff_pos**2).sum(axis=1)
    diff_neg_sq_sum = (diff_neg**2).sum(axis=1)
    diff_pos_Euclid = diff_pos_sq_sum ** .5
    diff_neg_Euclid = diff_neg_sq_sum ** .5
    dist = triplet_alpha - (diff_neg_sq_sum - diff_pos_sq_sum)
    return [dist, diff_pos_Euclid, diff_neg_Euclid]

##
input_var = T.matrix('inputs')
#target_var = T.bmatrix('targets')
net = {}
net['input'] = lasagne.layers.InputLayer((None, 3),input_var=input_var)
net['feature']  = Norm_L2_Layer(net['input'])

feature_embed = lasagne.layers.get_output(net['feature'])
triplet_alpha = 0.2
dist = get_dist_from_feature_embed(feature_embed)[0]
dist_thresh = dist*(dist>0)
    
############## cost with regularization ##############
cost = dist_thresh.sum()


train_fn = theano.function([input_var], [cost,feature_embed]+[dist,dist_thresh]+
                               get_dist_from_feature_embed(feature_embed))    

############## load the pretrained model ##############  

batch = np.random.randint(-2,2,(6,3)).astype(np.float32)
print batch
output = train_fn(batch)

dist_diff_batch = output[-3]
diff_pos_Euclid_batch = output[-2]
diff_neg_Euclid_batch = output[-1]
            