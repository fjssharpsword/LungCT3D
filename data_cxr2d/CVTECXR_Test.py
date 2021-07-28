import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import time
import random
from sklearn.model_selection import train_test_split
import sys
import torch.nn.functional as F
import scipy
import SimpleITK as sitk
import pydicom
from scipy import ndimage as ndi
from PIL import Image
import PIL.ImageOps 
from sklearn.utils import shuffle
import shutil
from matplotlib import pyplot as plt
import cv2
#define by myself
sys.path.append("..") 
from CXRAD.config import *
#from config import *
"""
Dataset: CVTE ChestXRay
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        for file_path in path_to_dataset_file:
            with open(file_path, "r") as f:
                for line in f: 
                    items = line.strip().split(',') 
                    image_name = os.path.join(path_to_img_dir, items[0])
                    if os.path.isfile(image_name) == True:
                        label = int(eval(items[1])) #eval for 
                        #if label ==0:  labels.append([0]) #normal
                        #else: labels.append([1]) #positive  
                        #if label != 0:
                        labels.append([label])
                        image_names.append(image_name) 

        self.image_names = image_names
        self.labels = labels

    def _transform_tensor(self, img):
        transform_seq = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
        return transform_seq(img)

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        try:
            image_name = self.image_names[index]
            image = Image.open(image_name).convert('RGB')
            image = self._transform_tensor(image)
            label = torch.as_tensor(self.labels[index], dtype=torch.long)
            
        except Exception as e:
            print("Unable to read file. %s" % e)
        
        return image, label

    def __len__(self):
        return len(self.image_names)


PATH_TO_IMAGES_DIR_CVTE = '/data/fjsdata/CVTEDR/images'
PATH_TO_TEST_FILE_CVTE = '/data/pycode/LungCT3D/data_cxr2d/cvte_test.txt'
def get_dataloader_CVTE(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_CVTE, path_to_dataset_file=[PATH_TO_TEST_FILE_CVTE])
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

if __name__ == "__main__":

    #for debug   
    data_loader_test = get_dataloader_CVTE(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image, label) in enumerate(data_loader_test):
        print(image.shape)
        print(label.shape)
        break



 
    
   
    
    
        
    
    
    
    