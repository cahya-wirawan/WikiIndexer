import unittest
from wit_dataset import WitDataset
from pathlib import Path


class TestWitDataset(unittest.TestCase):
    def test_wit_testfile(self):
        wd = WitDataset()
        wit_filename = Path("/mnt/mldata/data/WIT/test.tsv")
        wit_data = wd.read(wit_filename, chunksize=1000)
        print(len(wit_data))
        self.assertEqual(len(wit_data[0]), 681)
        self.assertEqual(len(wit_data[2]), 681)
        self.assertEqual(wit_data[3], 2123)
        self.assertEqual(wit_data[2][:5], [5, 5, 6, 6, 8])


if __name__ == '__main__':
    unittest.main()
