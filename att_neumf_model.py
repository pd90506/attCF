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
              e_dim=16, mlp_layer=[32], reg=0):
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

    
    mlp_user_embedding = keras.layers.Embedding(
        input_dim=num_users, output_dim=int(mlp_layer[0]/2),
        name='mlp_user_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    mlp_item_embedding = keras.layers.Embedding(
        input_dim=num_items, output_dim=int(mlp_layer[0]/2),
        name='mlp_item_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    aux_embedding = keras.layers.Embedding(
        input_dim=num_items, output_dim=int(mlp_layer[0]),
        name='aux_item_embedding',
        embeddings_initializer=init_normal(),
        embeddings_regularizer=keras.regularizers.l2(reg),
        input_length=1)

    # Flatten the output tensor
    gmf_user_latent = keras.layers.Flatten()(gmf_user_embedding(user_input))
    gmf_item_latent = keras.layers.Flatten()(gmf_item_embedding(item_input))
    mlp_user_latent = keras.layers.Flatten()(mlp_user_embedding(user_input))
    mlp_item_latent = keras.layers.Flatten()(mlp_item_embedding(item_input))
    aux_item_latent = keras.layers.Flatten()(aux_embedding(item_input))


    # Start gmf part
    gmf_vector = keras.layers.Multiply()([gmf_user_latent, gmf_item_latent])
    # Start mlp part
    mlp_vector = keras.layers.Concatenate(name='mlp_concatenation')(
        [mlp_user_latent, mlp_item_latent])

    # item vector feature extraction, split at the last layer
    for idx in range(1, num_layer-1):
        layer = keras.layers.Dense(
            units=mlp_layer[idx],
            activation='relu',
            kernel_initializer='lecun_uniform',
            kernel_regularizer=keras.regularizers.l2(reg),
            name='aux_item_layer_{:d}'.format(idx))
        aux_item_latent = layer(aux_item_latent)

    # aux_item_vector = keras.layers.Dense(
    #     units=num_tasks*mlp_layer[-1],
    #     activation='relu',
    #     kernel_initializer='lecun_uniform',
    #     kernel_regularizer=keras.regularizers.l2(reg),
    #     name='aux_item_vector')(aux_item_latent)
    # aux_item_vector = keras.layers.Reshape(
    #     (num_tasks, mlp_layer[-1]),
    #     name='multitask_item_vector')(aux_item_vector)

    # create multitask item output.
    item_feature_list = []  # all item features are stored here
    for idx in range(0, num_tasks):
        layer = keras.layers.Dense(
            units=mlp_layer[-1],
            activation='relu',
            kernel_initializer='lecun_uniform',
            kernel_regularizer=keras.regularizers.l2(reg),
            name='item_task_feature_{:d}'.format(idx))

        item_feature = layer(aux_item_latent)
        item_feature_list.append(item_feature)

    item_out_list = []   # all item outputs are stored here
    for idx in range(0, num_tasks):
        layer = keras.layers.Dense(
            units=1,
            activation='relu',
            kernel_initializer='lecun_uniform',
            kernel_regularizer=keras.regularizers.l2(reg),
            name='item_task_out_{:d}'.format(idx))

        item_task_output = layer(item_feature_list[idx])
        item_out_list.append(item_task_output)

    item_outputs = keras.layers.Concatenate()(item_out_list)


    # mlp vector feature extraction, no split
    for idx in range(1, num_layer):
        layer = keras.layers.Dense(
            units=mlp_layer[idx],
            activation='relu',
            kernel_initializer='lecun_uniform',
            kernel_regularizer=keras.regularizers.l2(reg),
            name='mlp_vector_layer_{:d}'.format(idx))
        mlp_vector = layer(mlp_vector)

    # Compute gmf attention scores use item_feature_list
    item_feature_matrix = keras.layers.Concatenate()(item_feature_list)
    item_feature_matrix = keras.layers.Reshape(
        (num_tasks, mlp_layer[-1]))(item_feature_matrix)
    gmf_weight_vector = keras.layers.Dot(axes=(-1, -1))(
        [item_feature_matrix, gmf_vector])

    gmf_weight_vector = keras.layers.Activation('softmax')(gmf_weight_vector)
    gmf_att_vector = keras.layers.Dot(axes=(-1, -2), name='gmf_attention_layer')(
        [gmf_weight_vector, item_feature_matrix])
    # Compute mlp attention scores use item_feature_list
    # mlp_weight_vector = keras.layers.Dot(axes=(-1, -1))(
    #     [item_feature_matrix, mlp_vector])

    # mlp_weight_vector = keras.layers.Activation('softmax')(mlp_weight_vector)
    # mlp_att_vector = keras.layers.Dot(axes=(-1, -2), name='mlp_attention_layer')(
    #     [mlp_weight_vector, item_feature_matrix])


    #  Concatenate gmf_vector and gmf_att_vector
    gmf_pred_vector = keras.layers.Concatenate()([gmf_vector, gmf_att_vector])
    #  Concatenate mlp_vector and mlp_att_vector
    # mlp_pred_vector = keras.layers.Concatenate()([mlp_vector, mlp_att_vector])

    # Concatenate gmf and mlp pred_vector
    pred_vector = keras.layers.Concatenate()([gmf_pred_vector, mlp_vector])

    prediction = keras.layers.Dense(
        units=1, activation='sigmoid',
        kernel_initializer='lecun_uniform',
        kernel_regularizer=keras.regularizers.l2(reg),
        name='prediction')(pred_vector)

    model = keras.models.Model(inputs=[user_input, item_input],
                               outputs=[prediction, item_outputs])


    return model
