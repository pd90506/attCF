import unittest
from genre_crossentropy import genre_crossentropy, genre_loss
import tensorflow as tf
import numpy as np


class LossTestCase(unittest.TestCase):
    """ Test loss functions"""
    def __init__(self, *args, **kwargs):
        super(LossTestCase, self).__init__(*args, **kwargs)
        # Parameter initiation
        self.batch_size = 100
        # Generate random model inputs
        self.y_true = tf.random_uniform(
            shape=(self.batch_size, 19), maxval=1, dtype=tf.float32)
        self.y_pred = tf.random_uniform(
            shape=(self.batch_size, 19), maxval=1, dtype=tf.float32)

    def test_genre_crossentropy(self):
        loss = genre_crossentropy(self.y_true, self.y_pred)
        print(loss.shape)
        self.assertEqual(loss.shape, self.batch_size)

    def test_genre_loss(self):
        loss = genre_loss(self.y_true, self.y_pred)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        loss_value = sess.run(loss)
        self.assertIsInstance(loss_value, np.float32)


if __name__ == '__main__':
    unittest.main()
