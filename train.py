# from model import get_model
# from att_mlp_model import get_model
from att_gmf_model import get_model
# from att_only import get_model
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import numpy as np
from time import time
from olddatasetclass import Dataset
from evaluate import evaluate_model
from tensorflow.keras.optimizers import Adam
from item_to_genre import item_to_genre
import pandas as pd
from aux_loss import aux_crossentropy_loss
import tensorflow as tf


def focal_loss(gamma=2., alpha=.25):
    '''
    Compatible with tensorflow backend
    '''
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_1 = tf.cast(pt_1, dtype='float32')
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt_0 = tf.cast(pt_0, dtype='float32')
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def nd_focal_loss(gamma=2, alpha=.25):
    ''' n dimensional version'''
    focal_loss_1d = focal_loss(gamma=gamma, alpha=alpha)

    def nd_focal_loss_fixed(y_true, y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        # assert(y_true.shape == y_pred.shape)
        dim = y_pred.shape[1]
        loss = 0
        for i in range(dim):
            # y_p = tf.layers.Flatten()(y_pred[:,i])
            # y_t = tf.layers.Flatten()(y_true[:,i])
            print(y_pred[:, i].shape)
            loss += focal_loss_1d(y_true[:, i], y_pred[:, i])
        return loss
    return nd_focal_loss_fixed



class Args(object):
    """Used to generate different sets of arguments"""
    def __init__(self):
        # default vaules
        self.path = 'Data/'
        self.dataset = 'ml-1m'
        self.epochs = 50
        self.batch_size = 256
        self.num_tasks = 18
        self.e_dim = 32
        self.mlp_layer = [256, 128,  64, 32]
        self.reg = [0, 0.0001, 0.0001]
        self.num_neg = 4
        self.lr = 0.001
        self.loss_weights = [0.5, 0.5]
        self.K = 10
        # self.learner = 'adam' 


def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [],[],[]
    # num_users = train.shape[0]
    # num_items = 1682 # 3952  ## TODO!
    # num_items = num_items
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while ((u,j) in train.keys()):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


def fit(args=Args()):
    # args = Args()
    result_out_file = 'outputs/%s_att_mlp_%s_top%d_edim%d_layer%s_%d.csv' %(args.dataset,
                                                                         args.loss_weights, args.K, args.e_dim,args.mlp_layer, time())
    topK = args.K
    evaluation_threads = 1  # mp.cpu_count()
    print("Att-Mul-MF arguments: %s " % (args))

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

    model.compile(optimizer=Adam(lr=args.lr), loss=['binary_crossentropy', focal_loss(gamma=5, alpha=0.1)], loss_weights=args.loss_weights)
    print(model.summary())

    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
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
        user_input, item_input, labels = get_train_instances(train, args.num_neg, num_items)
        dummy_genre = item_to_genre(item_input, data_size=args.dataset).values
        dummy_genre = np.nan_to_num(dummy_genre)
         # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         [np.array(labels), dummy_genre.astype(float)], # labels 
                         batch_size=args.batch_size, epochs=1, verbose=1, shuffle=True)
        t2 = time()

        
        # Evaluation
        if epoch %1 == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            
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
    
