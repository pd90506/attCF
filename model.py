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
    item_layers = [20, 18, 16]
    num_item_layer = len(item_layers)

    gmf_user_embedding = keras.layers.Embedding(
        input_dim=num_users, output_dim=int(e_dim),
        name='mlp_user_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    gmf_item_embedding = keras.layers.Embedding(
        input_dim=num_items, output_dim=int(e_dim),
        name='mlp_item_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    # Flatten the output tensor
    gmf_user_latent = keras.layers.Flatten()(gmf_user_embedding(user_input))
    gmf_item_latent = keras.layers.Flatten()(gmf_item_embedding(item_input))
    # GMF vector is used to make prediction
    gmf_vector = keras.layers.Multiply(name='gmf_layer')([gmf_user_latent, gmf_item_latent])

    # user vector feature extraction, doesn't split
    for idx in range(0, num_item_layer):
        layer = keras.layers.Dense(
            item_layers[idx],
            kernel_initializer=init_normal(),
            kernel_regularizer=keras.regularizers.l2(reg),
            activation='relu', name="user_layer%d" % idx)
        gmf_user_latent = layer(gmf_user_latent)

    # item vector feature extraction, split at the last layer
    for idx in range(0, num_item_layer-1):
        layer = keras.layers.Dense(
            item_layers[idx],
            kernel_initializer=init_normal(),
            kernel_regularizer=keras.regularizers.l2(reg),
            activation='relu', name="item_layer%d" % idx)
        gmf_item_latent = layer(gmf_item_latent)

    item_vector = keras.layers.Dense(
        units=num_tasks*item_layers[-1],
        activation='relu',
        kernel_initializer='lecun_uniform',
        kernel_regularizer=keras.regularizers.l2(reg),
        name='item_vector')(gmf_item_latent)
    item_vector = keras.layers.Reshape(
        (num_tasks, item_layers[-1]), name='multitask_vector')(item_vector)

    # weight_vector = keras.layers.Dot(axes=-1, normalize=True)(
    #     [item_vector, gmf_item_latent])
    # att_vector = keras.layers.Dot(axes=(-1, -2))([weight_vector, item_vector])
    m_gmf_vector = keras.layers.Multiply()([gmf_user_latent, item_vector])
    weight_vector = keras.layers.Dot(axes=-1, normalize=True)(
        [m_gmf_vector, gmf_user_latent])
    att_vector = keras.layers.Dot(axes=(-1, -2))([weight_vector, m_gmf_vector])
    att_vector = keras.layers.Reshape(
        (item_layers[-1],), name='attention_layer')(att_vector)

    # Concatenate att_vector and gmf_layer
    pred_vector = keras.layers.Concatenate()([gmf_vector, att_vector])

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
    aux_vector = keras.layers.Reshape((num_tasks,), name='aux_output')(aux_vector)

    model = keras.models.Model(inputs=[user_input, item_input],
                               outputs=[prediction, aux_vector])

    return model
