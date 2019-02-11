import unittest
from evaluate import string_to_array
from olddatasetclass_1 import Dataset
import numpy as np


class EvaluateTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(EvaluateTestCase, self).__init__(*args, **kwargs)
        self.string_array = "[1, 2, 3, 4, 5]"
        self.true_array = np.asarray([1, 2, 3, 4, 5])

    def test_string_to_array(self):
        target_array = string_to_array(self.string_array)
        print(len(target_array))
        self.assertTrue(np.array_equal(target_array, self.true_array))


class DatasetTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DatasetTestCase, self).__init__(*args, **kwargs)
        self.dataset = Dataset(path='Data/', size='ml-100k')
    
    def test_negatives(self):
        negatives = self.dataset.negatives
        print(negatives.head())
        print(negatives.shape)
        self.assertEqual(len(negatives.loc[0]), 99)
    
    def test_test_ratings(self):
        test_ratings = self.dataset.test_ratings
        print(test_ratings.head())
        print(test_ratings.shape)
        self.assertEqual(test_ratings.shape[0], 943)


if __name__ == '__main__':
    unittest.main()
