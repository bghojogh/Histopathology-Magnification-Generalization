import os
import sys

current_directory = sys.path[0]
above_directory, current_folder_name = os.path.split(current_directory)
sys.path.insert(1, above_directory)

import unittest
from unittest.mock import Mock
from unittest import TestCase, mock

from BreakHisOrganizer import BreakHisOrganizer
from ReinhardNormalizer import ReinhardNormalizer



@mock.patch("ReinhardNormalizer.ReinhardNormalizer")
def mock_ReinhardNormalizer(mock_class):

    print(mock_class.return_value)
            
class TestBreakHisOrganizer(TestCase):
    def test_if_constructible(self):
        break_his_organizer = BreakHisOrganizer('.', None, None, None, None, None)
        self.assertIsNotNone(break_his_organizer)
    
    def test_how_dataframe_is_loaded(self):
        dataset_root = 'D:/Datasets/masf_organized_breakhis_dataset/Best_after_normalization/'
        reinhard_normalizer = None
        normalization_ref_image = None
        path_extension = '*/*/*'
        image_extension = '.png'
        
        break_his_organizer_new_dataset = BreakHisOrganizer(dataset_root,
                                                                reinhard_normalizer,
                                                                normalization_ref_image,
                                                                path_extension,
                                                                image_extension)
        
        break_his_organizer_new_dataset.build_organized_dataframe()
            
    # def test_if_imwrite_is_called(self):
    #     with mock.patch('BreakHisOrganizer.cv2') as mocked_cv2:
    #         break_his_organizer = BreakHisOrganizer('.', None, None, None, None, None)
    #         break_his_organizer.save_dataset()
    #         mocked_cv2.imwrite.assert_called_once()
    
    # def test_if_Reinhard_is_injected_correctly(self):
    #     with mock.patch('ReinhardNormalizer.ReinhardNormalizer') as mocked_ReinhardNormalizer:
    #         break_his_organizer = BreakHisOrganizer('.', mocked_ReinhardNormalizer, None, None, None, None)
    #         break_his_organizer.save_dataset()
    #         mocked_ReinhardNormalizer.fit.assert_called_once()
    
        
    

if __name__=='__main__':
    unittest.main()