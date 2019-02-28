import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda


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
    layers = [64, 32, 16, 8] # dummy layers
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

    # Element-wise product
    mf_vector = keras.layers.Multiply()([user_latent, item_latent])

    # User feature and item multitask feature
    # user_feature = keras.layers.Dense(
    #     units=f_dim, kernel_regularizer=keras.regularizers.l2(reg),
    #     activation='relu', name='user_feature')(user_latent)

    # concatenate user latent and item latent, prepare for mlp part
    mlp_vector = keras.layers.Concatenate()([mlp_user_latent, mlp_item_latent])

    for idx in range(1, num_layer):
        layer = keras.layers.Dense(layers[idx], kernel_regularizer= keras.regularizers.l2(reg), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)
    # Create *num_tasks* parallel layers for all genres
    item_feature_list = []
    # Define the multitask output for auxiliary supervision
    item_output_list = []
    # Attention weights list
    att_weight_list = []
    # Construct multitask features and outputs
    for idx in range(num_tasks):
        item_feature = keras.layers.Dense(
            f_dim, activation='relu', name='item_feature%d' % idx)(mf_vector)
        item_feature_list.append(item_feature)
        item_output = keras.layers.Dense(
            1, activation='sigmoid', kernel_initializer='lecun_uniform',
            name='item_output%d' % idx)(item_feature)
        item_output_list.append(item_output)
        # Produce attention weights vector via inner products between
        # user feature vector and item feature vectors
        att_weight = keras.layers.Dot(axes=-1)([mlp_vector, item_feature])
        att_weight_list.append(att_weight)

    # Convert lists to vectors
    item_output_vector = keras.layers.Concatenate()(item_output_list)
    att_weight_vector = keras.layers.Concatenate()(att_weight_list)
    att_weight_vector = Lambda(normalize)(att_weight_vector)
    # TODO: WATCHOUT! which axis?
    item_feature_list = [keras.layers.Reshape(
        (f_dim, 1))(x) for x in item_feature_list]
    item_feature_tensor = keras.layers.concatenate(item_feature_list, axis=-1)

    # Weighted sum of different genres features based on attention weights
    weighted_sum = keras.layers.Dot(axes=-1)(
        [att_weight_vector, item_feature_tensor])

    weighted_sum = keras.layers.Flatten()(weighted_sum)
    # concatenate mlp part and weighted sum
    # final_layer = keras.layers.Concatenate()([mlp_vector, weighted_sum])

    # Final layer with sigmoid activation
    prediction = keras.layers.Dense(1, activation='sigmoid',
                                    kernel_initializer='lecun_uniform',
                                    name='prediction')(weighted_sum)

    # Construct model
    model = keras.models.Model(inputs=[user_input, item_input],
                               outputs=[prediction, item_output_vector])
    return model


