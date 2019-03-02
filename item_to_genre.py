import pandas as pd


def item_to_genre(item, data_size='ml-1m'):
    file_dir = 'movielens/' + data_size + '.movies'
    movies = pd.read_csv(file_dir, header=0, index_col=0)
    genre = movies.loc[item]
    return genre

def get_genre(data_size):
    file_dir = 'movielens/' + data_size + '.movies'
    movies = pd.read_csv(file_dir, header=0)
    items = movies.iloc[:, 0].values
    genres = movies.iloc[:, 1:].values
    return (items, genres)


get_genre('ml-100k')
