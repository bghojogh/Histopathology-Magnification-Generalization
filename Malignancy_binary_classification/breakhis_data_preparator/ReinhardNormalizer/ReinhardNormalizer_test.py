import unittest
from ReinhardNormalizer import ReinhardNormalizer

from unittest.mock import MagicMock
import cv2 as cv


class TestReinhardNormalizer(unittest.TestCase):
    def test_if_constructible(self):
        reinhard_normalizer = ReinhardNormalizer()
        self.assertIsNotNone(reinhard_normalizer)
    


if __name__ == '__main__':
    
    unittest.main()
    
    