import pandas as pd
import scipy.sparse as sp
import numpy as np

class Dataset(object):
    """ class docs"""
    def __init__(self, path, size='ml-1m'):
        trian_dir = path + size + '.train.rating'
        test_dir = path + size + '.test.rating'
        negatives_dir = path + size + '.test.negative'
        self.train_ratings = self.load_rating_file_as_matrix(trian_dir)
        self.test_ratings = self.load_train_ratings(test_dir)
        self.negatives = self.load_negatives(negatives_dir)
        # self.genre = self.load_genre(path + 'ml-1m.genre')
        # assert self.test_ratings.shape[0] == self.negatives.shape[0]

    def load_train_ratings(self, path):
        col_names = ['uid', 'mid', 'rating']
        train_ratings = pd.read_csv(path, sep=',', header=0, names=col_names)
        return train_ratings

    # def load_negatives(self, path):
    #     negatives = pd.read_csv(path, header=0, names=['uid', 'negatives'])
    #     negativeList = negatives['negatives'].apply(eval)
    #     return negativeList

    def load_negatives(self, path):
        negatives = pd.read_csv(path, sep=',', header=0)
        negatives['neg_list'] = negatives.iloc[:, 1].apply(eval)
        return negatives[['uid', 'neg_list']]

    # def load_genre(self, path):
    #     genre = pd.read_csv(path, header=0, names=['itemId', 'genre'])
    #     genreList = genre['genre'].values.tolist()
    #     return genreList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            next(f) # skip the header
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u) 
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            next(f) # skip the header
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat


if __name__ == '__main__':
    dataset = Dataset('Data/', 'ml-1m')
    x = dataset.train_ratings
    y = dataset.test_ratings
    z = dataset.negatives
    # a = dataset.genre
    print('pause')