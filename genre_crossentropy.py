import tensorflow.keras.backend as K
# import tensorflow.keras as keras


def genre_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def genre_loss(y_true, y_pred):
    return K.mean(genre_crossentropy(y_true, y_pred))