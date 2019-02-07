import unittest
import tensorflow as tf
from model import get_model


class ModelTestCase(unittest.TestCase):
    """ Test the model function"""

    def __init__(self, *args, **kwargs):
        super(ModelTestCase, self).__init__(*args, **kwargs)
        # Model param initializing
        self.num_users = 6040
        self.num_items = 3706
        self.num_tasks = 18
        self.e_dim = 16
        self.f_dim = 8
        self.reg = 0
        self.batch_size = 100
        # get model using the settings defined above
        self.model = get_model(
            self.num_users, self.num_items, self.num_tasks, e_dim=self.e_dim,
            f_dim=self.f_dim, reg=self.reg)

        # Generate random model inputs
        self.user_input = tf.random_uniform(
            shape=(self.batch_size, 1), maxval=self.num_users, dtype=tf.int32)
        self.item_input = tf.random_uniform(
            shape=(self.batch_size, 1), maxval=self.num_items, dtype=tf.int32)

    def test_prediction_dim(self):
        """ Test the dimension of model prediction. """
        model = self.model
        output = model([self.user_input, self.item_input])
        dim_sample = tf.random_uniform(shape=(self.batch_size, 1))
        self.assertEqual(output[0].shape, dim_sample.shape)

    def test_item_output_dim(self):
        """ Test the dimension of item output dimension. """
        model = self.model
        output = model([self.user_input, self.item_input])
        dim_sample = tf.random_uniform(shape=(self.batch_size, self.num_tasks))
        self.assertEqual(output[1].shape, dim_sample.shape)


if __name__ == '__main__':
    unittest.main()
