import pandas as pd
import random
import os


def sample_negative(ratings):
    """
    return 100 sampled negative items for each user
    args:
        ratings: ratings dataset, a dataframe
    """
    # user_pool = set(ratings['userId'].unique())
    item_pool = set(ratings['mid'].unique())

    interact_status = ratings.groupby('uid')['mid'].apply(set)\
        .reset_index().rename(columns={'mid': 'interacted_items'})
    interact_status['negative_items'] = interact_status['interacted_items']\
        .apply(lambda x: item_pool - x)
    interact_status['negative_samples'] = interact_status['negative_items']\
        .apply(lambda x: random.sample(x, 99))
    return interact_status[['uid', 'negative_samples']]


def split_train_test(ratings):
    """return training set and test set by loo"""
    ratings['date'] = pd.to_datetime(ratings['date'])
    ratings['rank_latest'] = ratings.groupby(['uid'])['date']\
        .rank(method='first', ascending=False)
    test = ratings[ratings['rank_latest'] == 1]
    test = test.sort_values('uid', ascending=True).reset_index(drop=True)
    train = ratings[ratings['rank_latest'] > 1]
#     assert train['uid'].nunique() == test['uid'].nunique()
    return train[['uid', 'mid', 'rating']], test[['uid', 'mid',
                                                  'rating']]


def sample_ciao(data_size='ciao', target_dir='Data'):
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    # define the file names of train, test and negatives
    train_file = target_dir + '/' + data_size + '.train.rating'
    test_file = target_dir + '/' + data_size + '.test.rating'
    test_negatives = target_dir + '/' + data_size + '.test.negative'
    # load the source data file
    source_dir = 'CiaoDVD/movie-ratings.txt'
    col_names = ['uid', 'mid', 'genre', 'rid', 'rating', 'date']
    ratings = pd.read_csv(source_dir, sep=',', header=None, names=col_names)
    negatives = sample_negative(ratings)
    train, test = split_train_test(ratings)
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    negatives.to_csv(test_negatives, index=False)
    export_genre(ratings)


def genre_to_list(genre):
    genre_list = [0] * 17
    genre_list[genre - 1] = 1
    return genre_list


def export_genre(movies):
    movies = movies.groupby('mid').first().reset_index()
    movies['genre_list'] = movies['genre'].apply(genre_to_list)
    new_df = movies['genre'].apply(genre_to_list)
    a = new_df.apply(lambda x: pd.Series(x))
    a['mid'] = movies['mid'].apply(lambda x: x)
    col_names = ['Action', 'Comedy',
                'Family', 'Drama', 'Horror', 'Fantasy',
                'Thriller', 'Martial', 'Musicals', 'War',
                'Westerns', 'Documentaries', 'Special', 'Sports', 'Cinema',
                'TV', 'Anime', 'mid']
    a.columns = col_names
    a = a[['mid', 'Action', 'Comedy',
                'Family', 'Drama', 'Horror', 'Fantasy',
                'Thriller', 'Martial', 'Musicals', 'War',
                'Westerns', 'Documentaries', 'Special', 'Sports', 'Cinema',
                'TV', 'Anime']]
    a.to_csv('movielens/ciao.movies', index=False)


if __name__ == "__main__":
    sample_ciao()