import pandas as pd
import numpy as np


def string_to_array(source_string):
    """
    Convert a quoted list to list
    """
    lst = eval(source_string)
    return lst


class Dataset(object):
    """ class docs"""
    def __init__(self, path, size='ml-100k'):
        trian_dir = path + size + '.train.rating'
        test_dir = path + size + '.test.rating'
        negatives_dir = path + size + '.test.negative'
        self.train_ratings = self.load_train_ratings(trian_dir)
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
        col_names = ['uid', 'negative_samples']
        negatives = pd.read_csv(path, sep=',', header=0,
                                names=col_names, index_col='uid')
        negatives['neg_array'] = negatives['negative_samples'].apply(
            string_to_array)
        return negatives['neg_array']

    # def load_genre(self, path):
    #     genre = pd.read_csv(path, header=0, names=['itemId', 'genre'])
    #     genreList = genre['genre'].values.tolist()
    #     return genreList


if __name__ == '__main__':
    dataset = Dataset('Data/', 'ml-1m')
    x = dataset.train_ratings
    y = dataset.test_ratings
    z = dataset.negatives
    # a = dataset.genre
    print('pause')