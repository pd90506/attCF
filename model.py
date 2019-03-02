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

    gmf_user_embedding = keras.layers.Embedding(
        input_dim=num_users, output_dim=int(e_dim),
        name='gmf_user_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    gmf_item_embedding = keras.layers.Embedding(
        input_dim=num_items, output_dim=int(e_dim),
        name='gmf_item_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    # Flatten the output tensor
    gmf_user_latent = keras.layers.Flatten()(gmf_user_embedding(user_input))
    gmf_item_latent = keras.layers.Flatten()(gmf_item_embedding(item_input))

    # Element-wise product of user and item latent, gmf
    gmf_vector = keras.layers.Multiply()([gmf_user_latent, gmf_item_latent])

    # item vector feature extraction, split at the last layer

    item_feature = keras.layers.Dense(
        units=num_tasks*f_dim,
        activation='relu',
        kernel_initializer='lecun_uniform',
        name='item_vector')(gmf_item_latent)
    item_feature = keras.layers.Reshape((num_tasks, f_dim))(
        item_feature)

    prediction = keras.layers.Dense(1, activation='sigmoid',
                                    kernel_initializer='lecun_uniform',
                                    name='prediction')(gmf_vector)

    # Auxiliary info output
    aux_vector = keras.layers.Dense(units=1,
                                    activation='sigmoid',
                                    kernel_initializer='lecun_uniform',
                                    name='aux_vector')(item_feature)
    aux_vector = keras.layers.Reshape((num_tasks,))(aux_vector)

    model = keras.models.Model(inputs=[user_input, item_input],
                               outputs=[prediction, aux_vector])
    aux_model = keras.models.Model(inputs=[item_input],
                                   outputs=[aux_vector])
    return (model, aux_model)
    # return model
