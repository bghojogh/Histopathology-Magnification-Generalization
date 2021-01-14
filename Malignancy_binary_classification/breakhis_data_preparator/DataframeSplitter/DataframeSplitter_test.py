import os
import sys

current_directory = sys.path[0]
above_directory, current_folder_name = os.path.split(current_directory)
sys.path.insert(1, above_directory)

import unittest

from DataframeSplitter import DataframeSplitter

import pandas as pd

class TestDataframeSplitter(unittest.TestCase):
    def test_if_constructible(self):
        dataframe_path = 'D:/Datasets/masf_organized_breakhis_dataset/Best_after_normalization/organized_dataframe.csv'
        df = pd.read_csv(dataframe_path)
        data_frame_splitter = DataframeSplitter([0.45,0.45,0.1], df)
        data_frame_splitter.split()
    

if __name__ == "__main__":
    unittest.main()