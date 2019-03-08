import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(path='Data/jester.train'):
    names = ['uid', 'mid', 'rating']
    df = pd.read_csv(path, sep=',', header=None, names=names)
    train, test = train_test_split(df, test_size=0.1)
    return (train, test)
