import pandas as pd
import numpy as np
import requests
import zipfile
from tqdm import tqdm
import math
import os
# from os.path import dirname, abspath
# from sklearn.model_selection import train_test_split


def download_ml(data_size):
    """
    Download movielens dataset of different sizes
    Args:
        data_size: a string that indicate the size of the dataset, e.g 'ml-1m',
        'ml-100k' etc.
    """
    data_urls = {
        'ml-100k': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
        'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'ml-20m': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
    }
    url = data_urls[data_size]
    file_name = 'temp/' + data_size + '.zip'
    # if the data directory already exist, return
    if os.path.isdir(data_size):
        return
    if not os.path.isdir('temp'):
        os.mkdir('temp')

    # if the directory doesn't exist, download and extract data
    with open(file_name, 'wb') as f:
        r = requests.get(url)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length'))
        block_size = 1024
        wrote = 0
        for data in tqdm(r.iter_content(block_size),
                         total=math.ceil(total_size/block_size),
                         unit='KB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
    # unzip to current directory
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall()


def create_csv_data(source_dir='ml-100k', target_dir='movielens'):
    """
    convert original data to csv file and movie to the 'movielens' folder
    args:
        source: source_dir .dat directory
        target: target_dir .csv directory
    """
    if source_dir == 'ml-100k':
        convert_100k(source_dir, target_dir)
    if source_dir == 'ml-1m':
        convert_1m(source_dir, target_dir)


def convert_100k(source_dir, target_dir):
    """
    convert ml-100k original dataset to csv files
    """
    # set input and output directories
    source_data = source_dir + '/u.data'
    source_movie = source_dir + '/u.item'
    target_data = target_dir + '/ml-100k.ratings'
    target_movie = target_dir + '/ml-100k.movies'

    # from source data to target data
    col_names = ['user_id', 'movie_id', 'rating', 'time_stamp']
    data = pd.read_csv(source_data, sep='\t', names=col_names)
    data.to_csv(target_data, index=False)

    # from source movie to target movie
    col_names = ['movie_id', 'title', 'release_date', 'video_release_date',
                 'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western']
    movies = pd.read_csv(source_movie, sep='|', names=col_names,
                         encoding='latin-1')
    movies = movies[['movie_id', 'Action', 'Adventure',
                     'Animation', 'Children', 'Comedy', 'Crime',
                     'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                     'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western']]
    movies.to_csv(target_movie, index=False)


def dat_to_csv(source, target):
    """
    convert .dat file to .csv file
    args:
        source: source .dat file path
        target: target .csv file path
    """
    df = pd.read_csv(source, sep='::', header=0)
    df.to_csv(target, index=False)


def convert_1m(source_dir, target_dir):
    # set input and output directories
    source_data = source_dir + '/ratings.dat'
    source_movie = source_dir + '/movies.dat'
    target_data = target_dir + '/ml-1m.ratings'
    target_movie = target_dir + '/ml-1m.movies'

    # from source data to target data
    col_names = ['user_id', 'movie_id', 'rating', 'time_stamp']
    data = pd.read_csv(source_data, sep='::', names=col_names)
    data.to_csv(target_data, index=False)

    # from source movies to target movies
    

def genre_to_int_list(genre_string):
    """
    Convert the list of genre names to a list of integer codes
    Args: 
        genre_string: a string of genres names.
    """
    GENRES = ('Action', 'Adventure', 'Animation', 'Children', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
              'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
              'War', 'Western')
    GENRES_LC = tuple((x.lower() for x in GENRES))
    # convert to lower case
    genre_string_lc = genre_string.lower()
    genre_list = []
    for idx in range(len(GENRES_LC)):
        if GENRES_LC[idx] in genre_string_lc:
            genre_list.append(idx)
    if len(genre_list) == 0:
        genre_list.append(-1)
    return genre_list



def genre_to_int_list(genre_string):
    """
    Convert the list of genre names to a list of integer codes
    Args:
        genre_string: a string of genres names.
    """
    GENRES = ('Action', 'Adventure', 'Animation', 'Children', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
              'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
              'War', 'Western')
    GENRES_LC = tuple((x.lower() for x in GENRES))
    # convert to lower case
    genre_string_lc = genre_string.lower()
    genre_list = []
    for idx in range(len(GENRES_LC)):
        if GENRES_LC[idx] in genre_string_lc:
            genre_list.append(idx)
    if len(genre_list) == 0:
        genre_list.append(-1)
    return genre_list


def loadMLData(file_dir, movie_dir):
    """
    Args:
        file_dir: the directory of the data file
        movie_dir: the directory of the movie title genre data file
    Load the MovieLens dataset, need to be a csv file
    """

    # read data from file and combine by merging,
    # select interested columns
    ml_rating = pd.read_csv(file_dir, header=0, \
                            names=['uid', 'mid', 'rating', 'timestamp'])
    mv_df = pd.read_csv(movie_dir, header=0, \
                            names=['mid', 'title', 'genre_string'])
    mv_df['genre'] = mv_df['genre_string'].apply(genre_to_single_int) # choose which kind of genre to output
    ml_rating = pd.merge(ml_rating, mv_df, on=['mid'], how='left')
    ml_rating = ml_rating.dropna()
    ml_rating = ml_rating[['uid', 'mid', 'rating', 'genre']]

    # Reindex 
    item_id = ml_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml_rating = pd.merge(ml_rating, item_id, on=['mid'], how='left')
    ml_rating = ml_rating.dropna()

    mv_df_new = pd.merge(mv_df, item_id, on=['mid'], how='left')
    mv_df_new = mv_df_new.dropna()
    mv_df_new = mv_df_new[['itemId', 'genre']].astype(int)

    user_id = ml_rating[['uid']].drop_duplicates()
    user_id['userId'] = np.arange(len(user_id))
    ml_rating = pd.merge(ml_rating, user_id, on=['uid'], how='left')
    
    
    #ml_rating['rating'] = ml_rating['rating'] # astype(int)

    ml_rating = ml_rating[['userId', 'itemId', 'rating', 'genre']]
    
    # Data prepared
    print('Range of userId is [{}, {}]'.format(ml_rating.userId.min(), \
          ml_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml_rating.itemId.min(), \
          ml_rating.itemId.max()))
    return(ml_rating, mv_df_new)



if __name__ == "__main__":
    create_csv_data(source_dir=)

