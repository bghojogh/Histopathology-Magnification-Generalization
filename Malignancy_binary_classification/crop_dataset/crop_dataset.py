# crop 227 x 227 pixels (similar to PACS) from 460 x 460 pixels:

import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os


path_dataset = 'C:\\Users\\benya\\Desktop\\my_PhD\\MAML\\datasets\\Pathology_dataset\\normalized_dataset\\'
path_save = 'C:\\Users\\benya\\Desktop\\my_PhD\\MAML\\datasets\\Pathology_dataset\\normalized_dataset_cropped\\'

folders = ['40', '100', '200', '400']
subfolders = ['benign_adenosis', 'benign_fibroadenoma', 'benign_phyllodes_tumor', 'benign_tubular_adenoma',
            'malignant_ductal_carcinoma', 'malignant_lobular_carcinoma', 'malignant_mucinous_carcinoma', 'malignant_papillary_carcinoma']

for folder in folders:
    for subfolder in subfolders:
        print('processing ' + folder + '/' + subfolder + '...')
        files = glob.glob(path_dataset + folder + '\\' + subfolder + "\\*")
        for file_address in files:
            name_of_image = file_address.split('\\')[-1]
            name_of_cancer = file_address.split('\\')[-2]
            name_of_magnification = file_address.split('\\')[-3]
            im = plt.imread(file_address)
            # plt.imshow(im)
            # plt.show()
            im_cropped = im[116:343, 116:343, :]
            # plt.imshow(im_cropped)
            # plt.show()
            address_save = path_save + name_of_magnification + '\\' + name_of_cancer + '\\'
            if not os.path.exists(address_save):
                os.makedirs(address_save)
            matplotlib.image.imsave(address_save + name_of_image, im_cropped)
        pass