from time import time
import scipy.io
import numpy as np


def process_netflix(source='netflix/combined_data_1.txt',
                    target='netflix/data.csv'):
    with open(source, "r") as f:
        line = f.readline()
        e = open(target, 'w')
        while line != '' and line is not None:
            if line[-2] == ':':
                mid = int(line[0:-2])  # determine the movie id
            else:
                arr = line.rstrip('\n').split(',')
                row = '{},{},{}\n'.format(arr[0], mid, arr[1])
                e.write(row)
            line = f.readline()
        e.close()


def process_netflix_genre(source='netflix/MovieGenreData.mat',
                          target='netflix/genre.csv'):
    data = scipy.io.loadmat(source)
    e = open(target, 'w')
    # number of specified genres
    num_genre = len(data['movie_genres_mapping'])

    genre = data['movie_genres_numeric']
    for idx in range(len(genre)):
        arr = np.zeros(num_genre, dtype=int)
        k = genre[idx][0]  # genre for index idx
        arr[k] = 1
        e.write(arr)
    
    e.close()
        


if __name__ == "__main__":
    t0 = time()
    # process_netflix()
    process_netflix_genre()
    print('Netflix data processing done! Time used: {:2.5f}'.format(time()-t0))
