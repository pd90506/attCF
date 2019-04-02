import tensorflow.keras as keras
import tensorflow.keras.backend as K
# import tensorflow as tf


def init_normal(shape=[0, 0.05], seed=None):
    mean, stddev = shape
    return keras.initializers.RandomNormal(
        mean=mean, stddev=stddev, seed=seed)


def normalize(tensor):
    K.l2_normalize(tensor)
    return tensor


def get_model(num_users, num_items, num_tasks,
              e_dim=16, reg=0, mlp_layer=[0] ):
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
    num_layer = len(mlp_layer)
    # Input variables
    user_input = keras.layers.Input(shape=(1,), dtype='int32',
                                    name='user_input')
    task_input = keras.layers.Input(shape=(num_tasks,), dtype='float',
                                    name='task_input')

    user_embedding = keras.layers.Embedding(
        input_dim=num_users, output_dim=int(mlp_layer[0]/2),
        name='user_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)
    task_layer = keras.layers.Dense(
        units=int(mlp_layer[0]/2), activation='softmax',
        kernel_initializer='lecun_uniform',
        kernel_regularizer=keras.regularizers.l2(reg),
        name='task_layer')

    # Flatten the output tensor
    user_latent = keras.layers.Flatten()(user_embedding(user_input))
    # task_latent = keras.layers.Flatten()(task_layer(task_input))
    task_latent = task_layer(task_input)
    # GMF layer
    mlp_vector = keras.layers.Concatenate()([user_latent, task_latent])
    for idx in range(1, num_layer):
        layer = keras.layers.Dense(
            units=mlp_layer[idx],
            activation='relu',
            kernel_initializer='lecun_uniform',
            kernel_regularizer=keras.regularizers.l2(reg),
            name='mlp_vector_layer_{:d}'.format(idx))
        mlp_vector = layer(mlp_vector)

    prediction = keras.layers.Dense(
        units=1, activation='sigmoid',
        kernel_initializer='lecun_uniform',
        kernel_regularizer=keras.regularizers.l2(reg),
        name='prediction')(mlp_vector)

    model = keras.models.Model(inputs=[user_input, task_input],
                               outputs=[prediction])

    return model
