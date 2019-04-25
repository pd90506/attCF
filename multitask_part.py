import tensorflow.keras as keras
import tensorflow.keras.backend as K


def init_normal(shape=[0, 0.05], seed=None):
    mean, stddev = shape
    return keras.initializers.RandomNormal(
        mean=mean, stddev=stddev, seed=seed)


def normalize(tensor):
    K.l2_normalize(tensor)
    return tensor


def get_model(num_users, num_items, num_tasks, e_dim=16, f_dim=8, reg=0):
    """
    This function is used to get the Att-Mul-MF model described
    in the paper.
    Args:
        :param num_users: number of users in the dataset
        :param num_items: number of items in the dataset
        :param num_tasks: number of tasks (item genres)
        :param e_dim: the embedding dimension
        :param f_dim: the preference feature space dimension
        :param reg: regularization coefficient
    """
    # Input variables
    user_input = keras.layers.Input(shape=(1,), dtype='int32',
                                    name='user_input')
    item_input = keras.layers.Input(shape=(1,), dtype='int32',
                                    name='item_input')

    user_embedding = keras.layers.Embedding(
        input_dim=num_users, output_dim=int(f_dim),
        name='user_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    item_embedding = keras.layers.Embedding(
        input_dim=num_items, output_dim=int(e_dim),
        name='item_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    # Flatten the output tensor
    user_latent = keras.layers.Flatten()(user_embedding(user_input))
    item_latent = keras.layers.Flatten()(item_embedding(item_input))

    # item vector feature extraction, split at the last layer
    item_vector = keras.layers.Dense(
        units=num_tasks*f_dim,
        activation='relu',
        kernel_initializer='lecun_uniform',
        kernel_regularizer=keras.regularizers.l2(reg),
        name='item_vector')(item_latent)
    item_vector = keras.layers.Reshape(
        (num_tasks, f_dim), name='multitask_vector')(item_vector)

    weight_vector = keras.layers.Dot(axes=-1, normalize=True)(
        [user_latent, item_vector])
    att_vector = keras.layers.Dot(axes=(-1, -2))([weight_vector, item_vector])
    att_vector = keras.layers.Flatten(name='attention_layer')(att_vector)

    # Concatenate att_vector and gmf_layer
    # pred_vector = keras.layers.Concatenate()([user_latent, att_vector])

    # prediction = keras.layers.Dense(
    #     units=1, activation='sigmoid',
    #     kernel_initializer='lecun_uniform',
    #     kernel_regularizer=keras.regularizers.l2(reg),
    #     name='prediction')(att_vector)

    pred_vector = keras.layers.Multiply()([user_latent, att_vector])
    prediction = keras.layers.Dense(
        units=1, activation='sigmoid',
        kernel_initializer='lecun_uniform',
        kernel_regularizer=keras.regularizers.l2(reg),
        name='prediction')(pred_vector)
    
    aux_vector = keras.layers.Dense(
        units=1, activation='sigmoid',
        kernel_initializer='lecun_uniform',
        kernel_regularizer=keras.regularizers.l2(reg),
        name='aux_vector')(item_vector)
    aux_vector = keras.layers.Flatten(name='aux_output')(aux_vector)

    model = keras.models.Model(inputs=[user_input, item_input],
                               outputs=[prediction, aux_vector])

    return model