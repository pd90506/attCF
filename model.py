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
    item_layers = [64, 32, 16, 8]
    num_item_layer = len(item_layers)

    # item_embedding = keras.layers.Embedding(
    #     input_dim=num_items, output_dim=item_layers[0], name='item_embedding',
    #     embeddings_initializer=init_normal(),
    #     embeddings_regularizer=keras.regularizers.l2(reg),
    #     input_length=1)

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

    # item vector feature extraction, split at the last layer
    for idx in range(1, num_item_layer-1):
        layer = keras.layers.Dense(
            item_layers[idx],
            kernel_regularizer=keras.regularizers.l2(reg),
            activation='relu', name="item_layer%d" % idx)
        mlp_item_latent = layer(mlp_item_latent)

    item_vector = keras.layers.Dense(
        units=num_tasks*item_layers[-1],
        activation='relu',
        kernel_initializer='lecun_uniform',
        name='item_vector')(mlp_item_latent)
    item_vector = keras.layers.Reshape((num_tasks, item_layers[-1]))(
        item_vector)

    weight_vector = keras.layers.Dot(axes=-1, normalize=True)(
        [item_vector, mlp_vector])
    att_vector = keras.layers.Dot(axes=(-1, -2))([weight_vector, item_vector])

    # Concatenate att_vector and mlp_vector
    pred_vector = keras.layers.Concatenate()([mlp_vector, att_vector])

    prediction = keras.layers.Dense(1, activation='sigmoid',
                                    kernel_initializer='lecun_uniform',
                                    name='prediction')(pred_vector)

    # Auxiliary info output
    aux_vector = keras.layers.Dense(units=1,
                                    activation='sigmoid',
                                    kernel_initializer='lecun_uniform',
                                    name='aux_vector')(item_vector)
    aux_vector = keras.layers.Reshape((num_tasks,))(aux_vector)

    model = keras.models.Model(inputs=[user_input, item_input],
                               outputs=[prediction, aux_vector])
    return model
