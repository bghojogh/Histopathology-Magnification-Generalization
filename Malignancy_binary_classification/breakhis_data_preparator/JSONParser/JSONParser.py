import json

class JSONParser(object):
    
    def __init__(self, file_path):
        self.__file_path = file_path
        self.setting = None
               
    def parse(self):
        with open(self.__file_path) as json_file:
            self.setting=json.load(json_file)
    
    def get_setting(self):
        return self.setting
    
    def get_item(self, parent, child):
        if self.setting is not None:
            return self.setting[parent][child]
    
    def get_dict(self, parent):
        if self.setting is not None:
            return self.setting[parent]