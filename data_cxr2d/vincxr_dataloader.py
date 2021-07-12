import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import random
import re
import sys
import scipy
import SimpleITK as sitk
import pydicom
from scipy import ndimage as ndi
import PIL.ImageOps 
from sklearn.utils import shuffle
import shutil
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2
from pycocotools import mask as coco_mask
#define by myself
"""
Dataset: VinBigData Chest X-ray Abnormalities Detection
https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data
1) 150,000 X-ray images with disease labels and bounding box
2) Label:['Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'No Finding']
"""
class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
        """
        data = pd.read_csv(path_to_dataset_file, sep=',')
        data = data.values #dataframe -> numpy
        images, labels, boxes = [], [], []
        for rec in data:
            images.append(rec[0].split(os.sep)[-1])
            labels.append(rec[8:].tolist()) #rec[1]-classname, rec[2]-labelid, rec[8:]=one-hot label
            boxes.append(rec[4:8])#rec[4-7],xmin,ymin,xmax,ymax

        self.image_dir = path_to_img_dir
        self.images = images
        self.labels = labels
        self.boxes = boxes

    def _transform_tensor(self, img):
        transform_seq = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
        return transform_seq(img)

    # segmentation
    #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
    def _get_seg(self, box, h, w):
        box = [box[0], box[1], box[2] - box[0], box[3] - box[1]] #x_min, y_min, width, height
        rles = coco_mask.frPyObjects(np.array([box], dtype=np.float), h, w)
        mask = coco_mask.decode(rles)
        return mask.squeeze()

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image = self.images[index]
        #image
        img_path = self.image_dir + image + '.jpeg'
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        image = self._transform_tensor(image)
        label = torch.as_tensor(self.labels[index], dtype=torch.float32)
        #mask
        box = self.boxes[index]
        mask = self._get_seg(box, height, width)
        mask = Image.fromarray(mask).resize((256, 256)) #numpy to pil image
        mask = torch.as_tensor(np.array(mask), dtype=torch.float32)
        #mask = transforms.ToTensor()(np.array(mask))
        
        return image, label, mask

    def __len__(self):
        return len(self.labels)

def get_train_dataloader_VIN(batch_size, shuffle, num_workers):
    vin_csv_file = '/data/pycode/LungCT3D/data_cxr2d/vincxr_train.csv'
    vin_image_dir = '/data/fjsdata/Vin-CXR/train_val_jpg/'
    dataset_train = DatasetGenerator(path_to_img_dir=vin_image_dir, path_to_dataset_file=vin_csv_file)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_test_dataloader_VIN(batch_size, shuffle, num_workers):
    vin_csv_file = '/data/pycode/LungCT3D/data_cxr2d/vincxr_test.csv'
    vin_image_dir = '/data/fjsdata/Vin-CXR/train_val_jpg/'
    dataset_test = DatasetGenerator(path_to_img_dir=vin_image_dir, path_to_dataset_file=vin_csv_file)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

def split_set():
    PATH_TO_IMAGES_DIR_Vin_Train = '/data/fjsdata/Vin-CXR/train/' #dicom file
    PATH_TO_FILES_DIR_Vin_Train = '/data/fjsdata/Vin-CXR/train.csv'

    imageIDs = []
    for root, dirs, files in os.walk(PATH_TO_IMAGES_DIR_Vin_Train):
        for file in files:
            imgID = os.path.splitext(file)[0]
            imageIDs.append(imgID)
    testIDs = random.sample(imageIDs, int(0.1*len(imageIDs)))
    trainIDs = list(set(imageIDs).difference(set(testIDs)))

    data = pd.read_csv(PATH_TO_FILES_DIR_Vin_Train, sep=',')
    data.fillna(0, inplace = True)
    data.loc[data["class_id"] == 14, ['x_max', 'y_max']] = 1.0
    data["class_id"] = data["class_id"] + 1
    data.loc[data["class_id"] == 15, ["class_id"]] = 0
    lbl_bin = pd.get_dummies(data["class_id"]) #dataframe
    data = pd.concat([data, lbl_bin], axis=1)
    print("\r VIN-CXR shape: {}".format(data.shape)) 
    print("\r VIN-CXR Columns: {}".format(data.columns))

    trainset, testset = [], []#pd.DataFrame(columns=data.columns), pd.DataFrame(columns=data.columns)
    for index, row in data.iterrows():
        if row['image_id'] in trainIDs:
            trainset.append(row.tolist())
        elif row['image_id'] in testIDs:
            testset.append(row.tolist())
        else:
            print('\r Image_ID {} is not exist'.format(row['image_id']))
        sys.stdout.write('\r index {} completed'.format(index+1))
        sys.stdout.flush()

    trainset = pd.DataFrame(trainset, columns=data.columns)
    print("\r trainset shape: {}".format(trainset.shape)) 
    print("\r trainset Columns: {}".format(trainset.columns))
    print("\r Num of disease: {}".format(trainset['class_id'].value_counts()) )
    trainset.to_csv('/data/pycode/LungCT3D/data_cxr2d/vincxr_train.csv', index=False, sep=',')#header=False

    testset = pd.DataFrame(testset, columns=data.columns)
    print("\r valset shape: {}".format(testset.shape)) 
    print("\r valset Columns: {}".format(testset.columns))
    print("\r Num of disease: {}".format(testset['class_id'].value_counts()) )
    testset.to_csv('/data/pycode/LungCT3D/data_cxr2d/vincxr_test.csv', index=False, sep=',')
    
if __name__ == "__main__":

    #split_set()

    #for debug   
    #data_loader = get_train_dataloader_VIN(batch_size=10, shuffle=True, num_workers=0)
    data_loader = get_test_dataloader_VIN(batch_size=10, shuffle=False, num_workers=0)
    for batch_idx, (image, label, mask) in enumerate(data_loader):
        print(label.shape)
        print(image.shape)
        print(mask.shape)
        break
  