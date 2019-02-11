import unittest
from item_to_genre import item_to_genre
# import numpy as np


class ItemToGenreTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ItemToGenreTestCase, self).__init__(*args, **kwargs)
        self.item = [1, 2, 3, 4, 5]

    def test_dim(self):
        genre = item_to_genre(self.item)
        print(genre)
        self.assertEqual(genre.shape, (5, 18))


if __name__ == '__main__':
    unittest.main()
