from model import get_model
import neumf
from jsdata import get_data
from time import time
from js_evaluate import evaluate_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np


class Args(object):
    """Used to generate different sets of arguments"""
    def __init__(self):
        # default vaules
        # self.model = 'model'
        self.model = 'neumf'
        self.path = 'Data/'
        self.dataset = 'jester'
        self.epochs = 20
        self.batch_size = 256
        self.num_tasks = 10  # need to be verified
        self.e_dim = 8
        self.mlp_layer = [64, 32, 16, 8]
        self.reg_layers = [0,0,0,0]
        self.reg = 0
        self.num_neg = 4
        self.lr = 0.001
        self.loss_weights = [1, 0.1]
        self.K = 10
        # self.learner = 'adam' 


def fit(args=Args()):
    result_out_file = 'outputs/%s_neumf_%s_top%d_edim%d_layer%s_%d.csv' %(args.dataset,
                                                                         args.loss_weights, args.K, args.e_dim, args.mlp_layer, time())
    print("Att-Mul-MF arguments: %s " % (args))

    # Load data
    t1 = time()
    if args.dataset == 'jester':
        num_users = 63979
        num_items = 151  # need modification
    else:
        raise Exception('wrong dataset size!!!')

    train, test = get_data()  # get train and test data
    print("Load data done [{:.1f} s]. #user={:d}, #item={:d}, #train={:d}, #test={:d}".format(
        time()-t1, num_users, num_items, len(train), len(test)))
    
    # Build model
    if args.model == 'model':
        model = get_model(num_users,
                        num_items,
                        num_tasks=args.num_tasks,
                        e_dim=args.e_dim,
                        mlp_layer=args.mlp_layer,
                        reg=args.reg)
        model.compile(optimizer=Adam(lr=args.lr), loss='binary_crossentropy', loss_weights=args.loss_weights)
    else:
        model = neumf.get_model(num_users, num_items, args.e_dim, args.mlp_layer, args.reg_layers, args.reg)
        model.compile(optimizer=Adam(lr=args.lr), loss='binary_crossentropy')

    print(model.summary())

    # get data
    user_input, item_input, labels = train['uid'].values.astype(int), train['mid'].values.astype(int), train['rating'].values.astype(int)
    test_user, test_item, test_rating = test['uid'].values.astype(int), test['mid'].values.astype(int), test['rating'].values.astype(int)

    # Init performance
    recall, auc = evaluate_model(model, test_user, test_item, test_rating)
    print('init recall is {:.4f}, auc is {:.4f}'.format(recall, auc))
    best_recall = recall
    best_auc = auc


    output = pd.DataFrame(columns=['recall','auc', 'loss'])
    loss = 1.0 ## TODO
    output.loc[0] = [recall, auc, loss]

    # Training model
    for epoch in range(int(args.epochs)):
        t1 = time()
        dummy = np.random.rand(len(train), args.num_tasks)
        if args.model == 'model':
            hist = model.fit([np.array(user_input), np.array(item_input)],  #input
                    [np.array(labels), dummy],  # labels 
                    batch_size=args.batch_size, epochs=1, verbose=1, shuffle=True)
        else:
            hist = model.fit([np.array(user_input), np.array(item_input)], #input
                    np.array(labels), # labels 
                    batch_size=args.batch_size, epochs=1, verbose=1, shuffle=True)     
        t2 = time()      
        # Evaluation
        if epoch % 1 == 0:
            recall, auc = evaluate_model(model, test_user, test_item, test_rating)
            loss = hist.history['loss'][0]
            print('Iteration %d [%.1f s]: recall = %.4f, auc= %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, recall, auc, loss, time()-t2))
            output.loc[epoch+1] = [recall, auc,  loss]
            if auc > best_recall:
                best_recall, best_auc, best_iter = recall, auc, epoch

    print("End. Best Iteration %d:  recall = %.4f, auc = %.4f. " %(best_iter, recall, auc))
    if args.out > 0:
        print("The best NeuMF model is saved to %s" %(model_out_file))

    output.to_csv(result_out_file, index=False)



if __name__ == '__main__':
    args1 = Args()
    fit(args1)
    # beta = np.linspace(0, 1, 11)
    # for b in beta:
    #     args = Args()s
    #     args.dataset = 'ml-1m'
    #     args.loss_weights = [1, b]
    #     fit(args)
    