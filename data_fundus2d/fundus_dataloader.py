import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import time
import random
import sys
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL.ImageOps
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

"""
Dataset: Indian Diabetic Retinopathy Image Dataset (IDRiD)
https://idrid.grand-challenge.org/
Link to access dataset: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
Data Descriptor: http://www.mdpi.com/2306-5729/3/3/25
First Results and Analysis: https://doi.org/10.1016/j.media.2019.101561
A. Localization: center pixel-locations of optic disc and fovea center for all 516 images;
B. Disease Grading: 516 images, 413(80%)images for training, 103(20%) images for test.
1) DR (diabetic retinopathy) grading: 0-no apparent retinopathy, 1-mild NPDR, 2-moderate NPDR, 3-Severe NPDR, 4-PDR
2) Risk of DME (diabetic macular edema): 0-no apparent EX(s), 1-Presence of EX(s) outside the radius of one disc diameter form the macula center,
                                        2-Presence of EX(s) within the radius of one disc diameter form the macula center.
C. Segmentation: 
1) 81 DR images, 54 for training and 27 for test.
2) types: optic disc(OD), microaneurysms(MA), soft exudates(SE), hard exudates(EX), hemorrhages(HE).
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_mask_dir):
        """
        Args:
            path_to_mask_dir: path to image directory.
            path_to_mask_dir: path to mask directory.
        """
        images = []
        masks = []
        for root, dirs, files in os.walk(path_to_img_dir):
            for file in files:
                images.append(os.path.join(path_to_img_dir + file))
                mask_file = os.path.splitext(file)[0]+'_OD.tif'
                masks.append(os.path.join(path_to_mask_dir + mask_file))

        self.images = images
        self.masks = masks
        self.transform_seq = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its mask
        """
        image = self.images[index]
        image = Image.open(image).convert('RGB')
        image = self.transform_seq(image)
        mask = self.masks[index]
        mask = cv2.imread(mask, cv2.COLOR_BGR2GRAY) #binary image
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask = np.where(mask>0, 1, 0)
        mask = torch.as_tensor(mask, dtype=torch.float32)  #.unsqueeze(0) #dtype=torch.long

        return image, mask

    def __len__(self):
        return len(self.images)


PATH_TO_IMAGES_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TrainingSet/'
PATH_TO_MASKS_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/OpticDisc/'
def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TRAIN, path_to_mask_dir=PATH_TO_MASKS_DIR_TRAIN)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

PATH_TO_IMAGES_DIR_TEST = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TestingSet/'
PATH_TO_MASKS_DIR_TEST = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TestingSet/OpticDisc/'
def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TEST, path_to_mask_dir=PATH_TO_MASKS_DIR_TEST)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

if __name__ == "__main__":
    #for debug   
    #SegHE_Annotation()
    dataloader_train = get_test_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (image, mask) in enumerate(dataloader_train):
        print(image.shape)
        print(mask.shape)
        break
