import unittest
from oldmodel import get_train_instances
from olddatasetclass import Dataset
from time import time


class OldmodelTestCase(unittest.TestCase):
    """oldmodel test case"""

    def __init__(self, *args, **kwargs):
        super(OldmodelTestCase, self).__init__(*args, **kwargs)
        path = 'Data/'
        size = 'ml-1m'
        self.dataset = Dataset(path, size=size)
        self.train = self.dataset.train_ratings
        self.num_negatives = 4

    def test_get_train_instances(self):
        time0 = time()
        train_samples = get_train_instances(self.train, self.num_negatives)
        print('Time wasted = {0:.3f}s'.format(float(time() - time0)))
        self.assertEqual(len(train_samples[0]), 994169 * 5)


if __name__ == '__main__':
    unittest.main()
