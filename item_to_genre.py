import pandas as pd


def item_to_genre(item, data_size='ml-1m'):
    file_dir = 'movielens/' + data_size + '.movies'
    movies = pd.read_csv(file_dir, header=0, index_col=0)
    genre = movies.iloc[item]
    return genre
