import tensorflow.keras as keras
import tensorflow.keras.backend as K

def aux_crossentropy_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)