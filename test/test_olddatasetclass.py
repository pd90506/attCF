import unittest
from olddatasetclass import Dataset


class OlddatasetclassTestCase(unittest.TestCase):
    """Test the old dataset class"""

    def __init__(self, *args, **kwargs):
        super(OlddatasetclassTestCase, self).__init__(*args, **kwargs)
        path = 'Data/'
        size = 'ml-1m'
        self.dataset = Dataset(path, size=size)

    def test_load_train_ratings(self):
        train_ratings = self.dataset.train_ratings
        # print(train_ratings.head())
        print(train_ratings.shape)
        self.assertEqual(train_ratings.shape, (994169, 3))

    def test_load_test_ratings(self):
        test_ratings = self.dataset.test_ratings
        # print(train_ratings.head())
        print(test_ratings.shape)
        self.assertEqual(test_ratings.shape, (6040, 3))

    def test_negatives(self):
        negatives = self.dataset.negatives
        # print(negative.head())
        self.assertEqual(negatives.shape, (6040, 2))
        lst = negatives.iloc[0, 1]
        self.assertEqual(len(lst), 99)


if __name__ == '__main__':
    unittest.main()
