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
import pickle
"""
Dataset: VinBigData Chest X-ray Abnormalities Detection
https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data
1) 150,000 X-ray images with disease labels and bounding box
2) Label:['Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'No Finding']
"""
#generate 
#https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#https://github.com/pytorch/vision
class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, bin_keys):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        total_csv_annotations = {}
        annotations = pd.read_csv(path_to_dataset_file, sep=',')
        annotations.fillna(0, inplace = True)
        annotations.loc[annotations["class_id"] == 14, ['x_max', 'y_max']] = 1.0
        annotations["class_id"] = annotations["class_id"] + 1
        annotations.loc[annotations["class_id"] == 15, ["class_id"]] = 0
        annotations = annotations[annotations.class_name!='No finding'].reset_index(drop=True) #remove samples "no finding"
        annotations = annotations.values #dataframe -> numpy
        for annotation in annotations:
            key = annotation[0].split(os.sep)[-1] 
            if key in bin_keys:
                value = np.array([annotation[1:]])
                if key in total_csv_annotations.keys():
                    total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
                else:
                    total_csv_annotations[key] = value

        self.image_dir = path_to_img_dir
        self.total_csv_annotations = total_csv_annotations
        self.total_keys = list(total_csv_annotations.keys())
        """
        #first split trainset and testset
        train_size = int(0.8 * len(self.total_keys))#8:2
        train_keys = random.sample(self.total_keys, train_size)
        test_keys = list(set(self.total_keys).difference(set(train_keys)))
        with open("/data/pycode/LungCT3D/data_cxr2d/trKeys.txt", "wb") as fp:   #Pickling
            pickle.dump(train_keys, fp)
        with open("/data/pycode/LungCT3D/data_cxr2d/teKeys.txt", "wb") as fp:   #Pickling
            pickle.dump(test_keys, fp)
        """
        #0 is background
        self.classname_to_id = {'Aortic enlargement':1, 'Atelectasis':2, 'Calcification':3, 'Cardiomegaly':4,
    		   	   'Consolidation':5, 'ILD':6, 'Infiltration':7, 'Lung Opacity':8, 'Nodule/Mass':9,
               	   'Other lesion':10, 'Pleural effusion':11, 'Pleural thickening':12, 'Pneumothorax':13, 'Pulmonary fibrosis':14}

    def _transform_tensor(self, img):
        transform_seq = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
        return transform_seq(img)

    # COCO： [x1,y1,w,h] 
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x, max_y]

    # compute area
    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x+1) * (max_y - min_y+1)

    # segmentation
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
        key = self.total_keys[index]
        #image
        img_path = self.image_dir + key + '.jpeg'
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        img = self._transform_tensor(img)
        #target
        shapes = self.total_csv_annotations[key]
        target = {}
        target["image_id"] = torch.as_tensor(index, dtype=torch.int64)
        target["iscrowd"] = torch.zeros((len(shapes),), dtype=torch.int64)
        boxes, labels, masks, areas = [],  [],  [],  []
        x_scale = 256/width
        y_scale = 256/height
        for shape in shapes:
            label = shape[0]
            points = shape[3:]
            points = [points[0]*x_scale, points[1]*y_scale, points[2]*x_scale, points[3]*y_scale] #resize box coordinates to (256,256)
            boxes.append(self._get_box(points))
            labels.append(int(self.classname_to_id[str(label)]))
            masks.append(self._get_seg(points, 256, 256))
            areas.append(self._get_area(points))
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) 
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)  #optional
        target["area"] = torch.as_tensor(areas, dtype=torch.float32) #optional
        
        return img, target

    def __len__(self):
        return len(self.total_keys)

def collate_fn(batch):
    return tuple(zip(*batch))
"""
def get_box_dataloader_VIN_None(batch_size, shuffle, num_workers):
    vin_csv_file = '/data/pycode/LungCT3D/data_cxr2d/train.csv'
    vin_image_dir = '/data/fjsdata/Vin-CXR/train_val_jpg/'
    dataset_box = DatasetGenerator(path_to_img_dir=vin_image_dir, path_to_dataset_file=vin_csv_file)

    train_size = int(0.8 * len(dataset_box))#8:2
    test_size = len(dataset_box) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_box, [train_size, test_size])
    data_loader_box_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    data_loader_box_test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    return data_loader_box_train, data_loader_box_test
"""
def get_box_dataloader_VIN(batch_size, shuffle, num_workers):
    vin_csv_file = '/data/pycode/LungCT3D/data_cxr2d/train.csv'
    vin_image_dir = '/data/fjsdata/Vin-CXR/train_val_jpg/'
  
    if shuffle==True: 
        with open("/data/pycode/LungCT3D/data_cxr2d/trKeys.txt", "rb") as fp:   # Unpickling
            key_subset = pickle.load(fp)
    else:
        with open("/data/pycode/LungCT3D/data_cxr2d/teKeys.txt", "rb") as fp:   # Unpickling
            key_subset = pickle.load(fp)

    dataset_box = DatasetGenerator(path_to_img_dir=vin_image_dir, path_to_dataset_file=vin_csv_file, bin_keys=key_subset)
    data_loader_box = DataLoader(dataset=dataset_box, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    return data_loader_box

if __name__ == "__main__":

    #for debug   
    data_loader_box = get_box_dataloader_VIN(batch_size=8, shuffle=True, num_workers=0)
    for batch_idx, (image, target) in enumerate(data_loader_box):
        print(len(image))
        print(len(target))
        break
