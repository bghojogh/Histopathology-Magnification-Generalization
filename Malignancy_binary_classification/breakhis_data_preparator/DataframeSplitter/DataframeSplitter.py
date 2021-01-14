import numpy as np
from JSONParser import JSONParser


class DataframeSplitter(object):
    def __init__(self, split_ratio, dataframe):
        self.test_prop, self.train_prop, self.validation_prop = split_ratio
        self.dataframe = dataframe
        self.json_parser = JSONParser.JSONParser('setting.json')
        self.json_parser.parse()
        self.label_to_int_dict = self.json_parser.get_dict('breakhis_label_to_integer') 
    
    def split(self):
        shuffled_dataframe = self.dataframe.sample(frac=1)
        reorganized = shuffled_dataframe.sort_values(by=['magnification'])
        magnification_list = reorganized.magnification.unique()
        for magnification in magnification_list:
            
            magnification_dataframe = reorganized[reorganized['magnification'] == magnification]
            test, train, validation = np.split(magnification_dataframe, [int(self.test_prop*len(magnification_dataframe)), int((self.test_prop+self.train_prop)*len(magnification_dataframe))])
            test = test.sort_values(by=['label'])
            train = train.sort_values(by=['label'])
            validation = validation.sort_values(by=['label'])
            self.log(str(magnification)+'_test.txt', test)
            self.log(str(magnification)+'_train.txt', train)
            self.log(str(magnification)+'_validation.txt', validation)
            

    def log(self, logger_name, dataframe):
        logger = open(logger_name,"w")
        for _, row in dataframe.iterrows():
            logger.write('/'.join(row.path.split('/')[-3:]) + ' ' + str(self.label_to_int_dict[row.label])+ "\n")
        logger.close()
                
                
                

                
            
            
            