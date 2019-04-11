# from model import get_model
# from att_mlp_model import get_model
# from att_gmf_model import get_model
# from att_neumf_model import get_model
# from att_cf import get_model
# from att_user_model import get_model
# import att_gmf_model
# import att_mlp_model
# from no_att_model import get_model
# from att_only import get_model
from att_cf import get_model
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import numpy as np
from time import time
from olddatasetclass import Dataset
from evaluate import evaluate_model
from tensorflow.keras.optimizers import Adam, SGD
from item_to_genre import item_to_genre
import pandas as pd
from aux_loss import aux_crossentropy_loss
from utils import get_train_instances


class Args(object):
    """Used to generate different sets of arguments"""
    def __init__(self):
        # default vaules
        self.model_name = 'att_user'
        self.path = 'Data/'
        self.dataset = 'ciao'
        self.epochs = 20
        self.batch_size = 256
        self.num_tasks = 17
        self.e_dim = 32
        self.mlp_layer = [256, 128, 64, 32]
        self.reg = 0
        self.num_neg = 4
        self.lr = 0.001
        self.loss_weights = [1, 1, 1]
        self.K = 10
        self.K2 = 20
        self.out = 1
        self.gmf_pretrain = ''
        self.mlp_pretrain = ''
        # self.gmf_pretrain = 'Pretrain/att_gmf_ml-1m_32_[256, 128, 64, 32]_1552197506.h5'
        # self.mlp_pretrain = 'Pretrain/att_mlp_ml-1m_32_[256, 128, 64, 32]_1552199426.h5'
        # self.learner = 'adam' 


def load_pretrain_model(model, gmf, mlp, num_layers, num_tasks):
    layer = [256,128,64,32]
    # GMF embeddings
    gmf_user_embeddings = gmf.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf.get_layer('item_embedding').get_weights()
    model.get_layer('gmf_user_embedding').set_weights(gmf_user_embeddings)
    model.get_layer('gmf_item_embedding').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_user_embedding').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_item_embedding').set_weights(mlp_item_embeddings)


    # Aux embeddings
    gmf_aux_embedding = gmf.get_layer('aux_item_embedding').get_weights()
    mlp_aux_embedding = mlp.get_layer('aux_item_embedding').get_weights()

    model.get_layer('aux_item_embedding').set_weights(gmf_aux_embedding)

    # aux layers
    for i in range(1, num_layers-1):
        gmf_layer_weights = gmf.get_layer('aux_item_layer_{:d}'.format(i)).get_weights()
        mlp_layer_weights = mlp.get_layer('aux_item_layer_{:d}'.format(i)).get_weights()

        model.get_layer('aux_item_layer_{:d}'.format(i)).set_weights(gmf_layer_weights)
    
    # aux multitask layers
    for i in range(0, num_tasks):
        gmf_layer_weights = gmf.get_layer('item_task_feature_{:d}'.format(i)).get_weights()
        mlp_layer_weights = mlp.get_layer('item_task_feature_{:d}'.format(i)).get_weights()

        model.get_layer('item_task_feature_{:d}'.format(i)).set_weights(gmf_layer_weights)
    
    # aux out layers
    for i in range(0, num_tasks):
        gmf_layer_weights = gmf.get_layer('item_task_out_{:d}'.format(i)).get_weights()
        mlp_layer_weights = mlp.get_layer('item_task_out_{:d}'.format(i)).get_weights()

        model.get_layer('item_task_out_{:d}'.format(i)).set_weights(gmf_layer_weights)


    # MLP vector layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp.get_layer('mlp_vector_layer_{:d}'.format(i)).get_weights()
        model.get_layer('mlp_vector_layer_{:d}'.format(i)).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf.get_layer('prediction').get_weights()
    mlp_prediction = mlp.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0][0:-layer[-1],:]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model


def fit(args=Args()):
    # args = Args()
    model_out_file = 'Pretrain/%s_%s_%d_%s_%d.h5' %(args.model_name, args.dataset, args.e_dim, args.mlp_layer, time())
    result_out_file = 'outputs/%s_%s_top%d_edim%d_layer%s_%d.csv' %(args.model_name, args.dataset,
                                                                         args.K, args.e_dim,args.mlp_layer, time())
    topK = args.K
    topK2 = args.K2
    print("%s arguments: %s " % (args.model_name, [args.dataset, args.e_dim, args.mlp_layer]))

    # Load data
    t1 = time()
    if args.dataset == 'ml-1m':
        num_users = 6040
        num_items = 3952  # need modification
    elif args.dataset == 'ml-100k':
        num_users = 943
        num_items = 1682
    elif args.dataset == 'ciao':
        num_users = 17615 + 1
        num_items = 16121 + 1
    else:
        raise Exception('wrong dataset size!!!')

    dataset = Dataset(args.path, args.dataset)
    train, testRatings, testNegatives = (dataset.train_ratings,
                                         dataset.test_ratings,
                                         dataset.negatives)

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time()-t1, num_users, num_items, train.shape[0],
             testRatings.shape[0]))

    # Build model, att model is a sub-routine, no need to train it
    model = get_model(num_users,
                      num_items,
                      num_tasks=args.num_tasks,
                      e_dim=args.e_dim,
                      mlp_layer=args.mlp_layer,
                      reg=args.reg)

    model.compile(optimizer=Adam(lr=args.lr), loss='binary_crossentropy', loss_weights=args.loss_weights)
    print(model.summary())

    # Load pretrain model
    if args.gmf_pretrain != '' and args.mlp_layer != '':
        gmf = att_gmf_model.get_model(num_users, num_items, args.num_tasks, e_dim=args.e_dim, mlp_layer=args.mlp_layer, reg=args.reg)
        gmf.load_weights(args.gmf_pretrain)
        mlp = att_mlp_model.get_model(num_users, num_items, args.num_tasks, e_dim=args.e_dim, mlp_layer=args.mlp_layer, reg=args.reg)
        mlp.load_weights(args.mlp_pretrain)
        model = load_pretrain_model(model, gmf, mlp, len(args.mlp_layer),args.num_tasks)
        print("Load pretrained GMF (%s) and MLP (%s) models done. " %(args.gmf_pretrain, args.mlp_pretrain))

    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
    (hits2, ndcgs2) = evaluate_model(model, testRatings, testNegatives, topK2)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    #dummy_genre = np.random.randn(4970845, args.num_tasks)
    
    # save Hit ratio and ndcg, loss
    output = pd.DataFrame(columns=['hr', 'ndcg', 'loss'])
    loss = 1.0 ## TODO
    output.loc[0] = [hr, ndcg, loss]

    # Training model
    for epoch in range(int(args.epochs)):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, args.num_neg, num_items, args.num_neg)
        dummy_genre = item_to_genre(item_input, data_size=args.dataset).values
        dummy_genre = np.nan_to_num(dummy_genre)
        dummy_sim = np.zeros(len(user_input))
         # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         [np.array(labels), dummy_genre, dummy_sim], # labels 
                         batch_size=args.batch_size, epochs=1, verbose=1, shuffle=True)
        t2 = time()

        
        # Evaluation
        if epoch %1 == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
            (hits2, ndcgs2) = evaluate_model(model, testRatings, testNegatives, topK2)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            hr2, ndcg2 = np.array(hits2).mean(), np.array(ndcgs2).mean()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            print('K2 Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr2, ndcg2, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

            output.loc[epoch+1] = [hr, ndcg, loss]

            # att_vector_of_0 = att_model.predict([np.array([0]), np.array([0])])
            # aux_vector_of_0 = aux_model.predict([np.array([0])])
            # print('The attention vector for user 0 on movie 0 is:\n {}'.format(att_vector_of_0))
            # print('The output aux vector for movie 0 is:\n {}'.format(aux_vector_of_0))
            # print('The true genre info for movie 0 is:\n {}'.format(item_to_genre(0).values))
            # if epoch %3 == 0:
            #     input()
    
    output.to_csv(result_out_file, index=False)
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    


if __name__ == '__main__':
    args1 = Args()
    fit(args1)
    # beta = np.linspace(0, 1, 11)
    # for b in beta:
    #     args = Args()s
    #     args.dataset = 'ml-1m'
    #     args.loss_weights = [1, b]
    #     fit(args)
    
