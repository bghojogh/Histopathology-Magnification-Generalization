import unittest

from JSONParser import JSONParser

class TestJSONParser(unittest.TestCase):
    def test_JSONParser_constructible(self):
        path = 'c:/dummy_path/'
        json_parser = JSONParser(path)
        self.assertIsNotNone(json_parser)
    
    def test_if_json_is_loaded(self):
        path = './JSONParser/test.json'
        json_parser = JSONParser(path)
        json_parser.parse()
        setting = json_parser.get_setting()
        self.assertIsNotNone(setting,"There is a json file and \
            have been loaded")
        
    def test_if_item_can_be_retrieved(self):
        path = './JSONParser/test.json'
        json_parser = JSONParser(path)
        json_parser.parse()
        item = json_parser.get_item('parent', 'child')
        self.assertEqual(item, 'item', "The item has been retrieved succefully!")
    
    def test_if_can_work_with_nested_dict(self):
        path = './JSONParser/nested.json'
        json_parser = JSONParser(path)
        json_parser.parse()
        item = json_parser.get_item('breakhis_label_to_integer', 'benign_tubular_adenoma')
        
        self.assertEqual(item, 4, "The item has been retrieved succefully!")
    
    def test_if_can_get_a_dict(self):
        path = './JSONParser/nested.json'
        json_parser = JSONParser(path)
        json_parser.parse()
        dict_retrieved = json_parser.get_dict('breakhis_label_to_integer')
        item = dict_retrieved['benign_tubular_adenoma']
        self.assertEqual(item, 4, "The item has been retrieved succefully!")
        


if __name__=='__main__':
    unittest.main()