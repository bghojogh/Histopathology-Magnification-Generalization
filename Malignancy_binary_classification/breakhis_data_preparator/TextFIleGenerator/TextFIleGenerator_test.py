import unittest
from TextFileGenerator import TextFileGenerator

class TestTextFileGenerator(unittest.TestCase):
    def test_if_constructible(self):
        csv_path = 'D:/Datasets/square_breast/dataset_dataframe.csv'
        text_file_generator = TextFileGenerator()
        