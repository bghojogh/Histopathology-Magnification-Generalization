import os
import sys
from os import path

from glob import glob
from pathlib import Path
import shutil

import pandas as pd

import cv2

from StainingUtils import StainingUtils


class BreakHisOrganizer(object):
    def __init__(self, dataset_root, normalizer, normalization_reference_image, path_extension, image_extension):
        self.normalizer = normalizer
        self.dataset_root = dataset_root
        self.normalization_reference_image = normalization_reference_image
        self.path_extension = path_extension
        self.img_extension = image_extension
        self.dataframe = None
        
    def build_dataframe(self):
        images_path = [img_file.replace("\\", "/") for img_file in glob(self.dataset_root + self.path_extension + self.img_extension)]
        """
        you will face with some magic numbers. These are specific to BreakHis dataset. The class has been designed for working with this dataset.
        """
        images_magnification = [(image_path.split('/')[-1]).split('-')[-2] for image_path in images_path]
        images_class = [image_path.split('/')[3]+'_'+image_path.split('/')[5] for image_path in images_path]
        slides_name = [(image_path.split('/')[6]).split('_')[-1] for image_path in images_path]
        self.dataframe = pd.DataFrame(list(zip(images_path, images_magnification, images_class, slides_name)), 
                columns =['path', 'magnification', 'label', 'slide_name'])
    
    def build_organized_dataframe(self):
        images_path = [img_file.replace("\\", "/") for img_file in glob(self.dataset_root + self.path_extension + self.img_extension)]
        """
        you will face with some magic numbers. These are specific to BreakHis dataset. The class has been designed for working with this dataset.
        """
        images_magnification = [(image_path.split('/')[-1]).split('-')[-2] for image_path in images_path]
        images_class = [image_path.split('/')[5] for image_path in images_path]
        slides_name = ['-'.join((((image_path.split('/')[6]).split('_')[-1]).split('-')[:-2])) for image_path in images_path]
        self.dataframe = pd.DataFrame(list(zip(images_path, images_magnification, images_class, slides_name)), 
                columns =['path', 'magnification', 'label', 'slide_name'])
        
    def set_dataframe(self, dataframe):
        self.dataframe = dataframe
        
    def write_dataframe(self, dataframe_name):
        self.dataframe.to_csv (self.dataset_root + dataframe_name + '.csv', index=False)
    
    def __create_new_dataset_directory(self, new_dataset_name):
        Path(self.dataset_root + new_dataset_name).mkdir(parents=True, exist_ok=True)
        return self.dataset_root + new_dataset_name
        
    def save_dataset(self, new_dataset_name, stain_normalization=True, load_from_csv=False, dataframe_name='dataset_dataframe.csv'):
        self.normalizer.fit(self.normalization_reference_image)
        new_dataset_directory = self.__create_new_dataset_directory(new_dataset_name)
        
        if load_from_csv: 
            df = pd.read_csv(self.dataset_root + dataframe_name)
        else:
            df= self.dataframe
        
        for _, row in df.iterrows():
            image_name = row.path.split('/')[-1]
            current_directory = new_dataset_directory + '/' + str(row.magnification) + '/' + str(row.label) + '/'
            Path(current_directory).mkdir(parents=True, exist_ok=True)
            if not path.exists(current_directory+image_name):
                if stain_normalization:
                    source_image = StainingUtils.read_image(row.path)
                    output_image = self.normalizer.transform(source_image)
                    cv2.imwrite(current_directory+image_name, output_image)
                else:
                    shutil.copy(row.path, current_directory)
        