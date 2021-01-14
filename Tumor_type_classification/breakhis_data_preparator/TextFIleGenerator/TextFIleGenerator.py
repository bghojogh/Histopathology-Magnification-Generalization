import numpy as np
import pandas as pd

class TextFileGenerator(object):
    def __init(self, csv_path):
        data_table = pd.read_csv(csv_path)
    
    def write(self, save_path):
        if data_table:
            