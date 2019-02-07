from model import get_model
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import numpy as np
from time import time
from olddatasetclass import Dataset
from evaluate_legacy_1 import evaluate_model
from tensorflow.keras.optimizers import Adam
from item_to_genre import item_to_genre


class Args(object):
    """Used to generate different sets of arguments"""
    def __init__(self):
        self.path = 'Data/'
        self.dataset = 'ml-1m'
        self.epochs = 20
        self.batch_size = 256
        self.num_tasks = 18
        self.e_dim = 16
        self.f_dim = 8
        self.reg = 0
        self.num_neg = 4
        self.lr = 0.001
        # self.learner = 'adam' 


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
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
    args = Args()
    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("Att-Mul-MF arguments: %s " % (args))

    # Load data
    t1 = time()
    if args.dataset == 'ml-1m':
        num_users = 6040
        num_items = 3706  # need modification
    elif args.dataset == 'ml-100k':
        num_users = 943
        num_items = 1682
    else:
        raise Exception('wrong dataset size!!!')

    dataset = Dataset(args.path, args.dataset)
    train, testRatings, testNegatives = (dataset.train_ratings,
                                         dataset.test_ratings,
                                         dataset.negatives)

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time()-t1, num_users, num_items, train.shape[0],
             testRatings.shape[0]))

    # Build model
    model = get_model(num_users,
                      num_items,
                      num_tasks=args.num_tasks,
                      e_dim=args.e_dim,
                      f_dim=args.f_dim,
                      reg=args.reg)

    model.compile(optimizer=Adam(lr=args.lr), loss='binary_crossentropy',
                  loss_weights=[0.8, 0.2])

    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK,
                                   evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    # dummy_genre = np.random.randn(4970845, args.num_tasks)

    # Training model
    for epoch in range(int(args.epochs)):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, args.num_neg)
        dummy_genre = item_to_genre(item_input)
         # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         [np.array(labels), dummy_genre], # labels 
                         batch_size=args.batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %1 == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
    
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))


if __name__ == '__main__':
    fit()
