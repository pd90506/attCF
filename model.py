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

    # Embedding layer
    layers = [64, 32, 16, 8]  # dummy layers
    num_layer = len(layers)

    user_embedding = keras.layers.Embedding(
        input_dim=num_users, output_dim=e_dim, name='user_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    item_embedding = keras.layers.Embedding(
        input_dim=num_items, output_dim=e_dim, name='item_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    mlp_user_embedding = keras.layers.Embedding(
        input_dim=num_users, output_dim=int(layers[0]/2),
        name='mlp_user_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    mlp_item_embedding = keras.layers.Embedding(
        input_dim=num_items, output_dim=int(layers[0]/2),
        name='mlp_item_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    # Flatten the output tensor
    user_latent = keras.layers.Flatten()(user_embedding(user_input))
    item_latent = keras.layers.Flatten()(item_embedding(item_input))
    mlp_user_latent = keras.layers.Flatten()(mlp_user_embedding(user_input))
    mlp_item_latent = keras.layers.Flatten()(mlp_item_embedding(item_input))

    # concatenate user latent and item latent, prepare for mlp part
    mlp_vector = keras.layers.Concatenate()([mlp_user_latent, mlp_item_latent])

    for idx in range(1, num_layer):
        layer = keras.layers.Dense(
            layers[idx],
            kernel_regularizer=keras.regularizers.l2(reg),
            activation='relu', name="mlp_layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    # Element-wise product
    mf_vector = keras.layers.Multiply()([user_latent, item_latent])
    mf_vector = keras.layers.Dense(units=f_dim*num_tasks,
                                   activation='relu',
                                   kernel_initializer='lecun_uniform',
                                   name='mf_vector')(mf_vector)
    mf_vector = keras.layers.Reshape((num_tasks, f_dim))(mf_vector)

    weight_vector = keras.layers.Dot(axes=-1, normalize=True)(
        [mf_vector, mlp_vector])
    att_vector = keras.layers.Dot(axes=(-1, -2))([weight_vector, mf_vector])

    prediction = keras.layers.Dense(1, activation='sigmoid',
                                    kernel_initializer='lecun_uniform',
                                    name='prediction')(att_vector)

    # Auxiliary info output
    aux_vector = keras.layers.Dense(units=1,
                                    activation='sigmoid',
                                    kernel_initializer='lecun_uniform',
                                    name='aux_vector')(mf_vector)
    aux_vector = keras.layers.Reshape((num_tasks,))(aux_vector)

    model = keras.models.Model(inputs=[user_input, item_input],
                               outputs=[prediction, aux_vector])
    return model
