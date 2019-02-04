import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, Lambda, Activation
from tensorflow.keras.layers import Embedding, Input, Dense, Multiply, Reshape, Flatten, Concatenate
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from evaluate_legacy import evaluate_model
from olddatasetclass import Dataset
from time import time
import multiprocessing as mp
import sys


######Init args######
class Args(object):
    """A simulator of parser in jupyter notebook"""
    def __init__(self):
        self.path = 'Data/'
        self.dataset = 'ml-1m'
        self.epochs = 20
        self.batch_size = 256
        self.num_factors = 8
        self.layers = '[64,32,16,8]'
        self.reg_mf = '[0,0]'
        self.reg_layers = '[0,0,0,0]'
        self.num_neg = 4
        self.lr = 0.001
        self.learner = 'adam'
        self.verbose = 0
        self.out = 1
        self.mf_pretrain = ''
        self.mlp_pretrain= ''

def init_normal(shape=[0,0.05], seed=None):
    mean, stddev = shape
    return initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg_mf[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg_mf[1]), input_length=1)  

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'mlp_user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg_layers[0]), input_length=1)  
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    # Element-wise product of user and item embeddings 
    mul = Multiply()
    mf_vector = mul([mf_user_latent, mf_item_latent])

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    # The 0-th layer is the concatenation of embedding layers
    concat = Concatenate()
    mlp_vector = concat([mlp_user_latent, mlp_item_latent])
    
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = concat([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=prediction)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_items = len(train['mid'].unique())
    for _, row in train.iterrows():
        # positive instance
        u = row['uid']
        i = row['mid']
        user_input.append(int(u))
        item_input.append(int(i))
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            # while ((u,j) in train.keys()):
            #     j = np.random.randint(num_items)
            user_input.append(int(u))
            item_input.append(int(j))
            labels.append(0)
    return user_input, item_input, labels

def fit():
    #args = parse_args()
    args = Args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = eval(args.reg_mf)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    # Override args
    # args.dataset = name_data
    # args.batch_size = batch_size
            
    topK = 10
    evaluation_threads = 1#mp.cpu_count()
    print("NeuMF arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.h5' %(args.dataset, mf_dim, args.layers, time())
    result_out_file = 'outputs/%s_NeuMF_%d_%s_%d.csv' %(args.dataset, mf_dim, args.layers, time())

     # Loading data
    t1 = time()
    if args.dataset=='ml-1m':
        num_users = 6040
        num_items = 3706 # need modification
    elif args.dataset=='ml-100k':
        num_users = 943
        num_items = 1682
    else:
        raise Exception('wrong dataset size!!!')   

    dataset = Dataset(args.path, args.dataset)
    train, testRatings, testNegatives = dataset.train_ratings, dataset.test_ratings, dataset.negatives

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
        %(time()-t1, num_users, num_items, train.shape[0], testRatings.shape[0]))
    
        # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    
    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = GMF.get_model(num_users,num_items,mf_dim)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLP.get_model(num_users,num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))
        
    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True) 

    # save Hit ratio and ndcg, loss
    output = pd.DataFrame(columns=['hr', 'ndcg'])
    output.loc[0] = [hr, ndcg]
    


    # Training model
    for epoch in range(int(num_epochs)):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
         # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %1 == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            output.loc[epoch+1] = [hr, ndcg]
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)
    
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best NeuMF model is saved to %s" %(model_out_file))

    output.to_csv(result_out_file)
    return([best_iter, best_hr, best_ndcg])


if __name__ == '__main__':
    fit()