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
import sys
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
#define by myself
sys.path.append("..") 
from VR3D.config import *
#from config import *
"""
Dataset:LIDC-IDRI
https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_dataset_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        vol_imgs = []
        vol_masks = []
        vol_labels = []
        datas = pd.read_csv(path_to_dataset_file, names =['vol_id','nod_id','slice_id','img','mask','mali_label','cancer_flag','clean_flag'], sep=',', header=None)
        vol_list = datas['vol_id'].unique().tolist() #first col
        for vol in vol_list:
            vol_df = datas[datas['vol_id']==vol] 
            nod_list = vol_df['nod_id'].unique().tolist()
            for nod in nod_list:
                nod_np = vol_df[vol_df['nod_id']==nod].values #dataframe->numpy
                slice_names = []
                mask_names = []
                mali_labels = []
                for nod_line in nod_np:
                    slice_name = os.path.join(PATH_TO_IMAGES_DIR, nod_line[0], nod_line[3]+'.npy') 
                    slice_names.append(slice_name)
                    mask_name = os.path.join(PATH_TO_MASKS_DIR, nod_line[0], nod_line[4]+'.npy') 
                    mask_names.append(mask_name)
                    mali_label = nod_line[5] 
                    mali_labels.append(mali_label)
                vol_imgs.append(slice_names)
                vol_masks.append(mask_names)
                assert len(set(mali_labels))== 1 #equal label for the same nodule
                vol_labels.append(list(set(mali_labels)))
                
        self.vol_imgs = vol_imgs
        self.vol_masks = vol_masks
        self.vol_labels = vol_labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image:NxCxDxHxW
            mask:NxCxDxHxW
            label:1
        """
        #get image
        vol_imgs =  self.vol_imgs[index]
        ts_imgs = torch.FloatTensor()
        for img in vol_imgs:
            img = np.load(img)
            img = torch.as_tensor(img, dtype=torch.float32)
            ts_imgs = torch.cat((ts_imgs, img.unsqueeze(0)), 0)
        ts_imgs = zoom(ts_imgs, (config['VOL_DIMS']/ts_imgs.shape[0], 0.125, 0.125), order=1)
        ts_imgs = torch.as_tensor(ts_imgs, dtype=torch.float32)
        ts_imgs = ts_imgs.unsqueeze(0) #1x32x64x64
        #get masks
        vol_masks = self.vol_masks[index]
        ts_masks = torch.FloatTensor()
        for mask in vol_masks:
            mask = np.load(mask)
            mask = torch.as_tensor(mask, dtype=torch.float32)
            ts_masks = torch.cat((ts_masks, mask.unsqueeze(0)), 0)
        ts_masks = zoom(ts_masks, (config['VOL_DIMS']/ts_masks.shape[0], 0.125, 0.125), order=1)
        ts_masks = torch.as_tensor(ts_masks, dtype=torch.float32)
        ts_masks = ts_masks.unsqueeze(0)#1x32x64x64
        #get label
        ts_label = torch.as_tensor(self.vol_labels[index], dtype=torch.float32)
        return ts_imgs, ts_masks, ts_label

    def __len__(self):
        return len(self.vol_labels)

PATH_TO_TRAIN_FILE = '/data/pycode/VR3D/datasets/LIDC_VR_Train.txt'
def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_dataset_file=PATH_TO_TRAIN_FILE)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

PATH_TO_TEST_FILE = '/data/pycode/VR3D/datasets/LIDC_VR_Test.txt'
def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_dataset_file=PATH_TO_TEST_FILE)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

if __name__ == "__main__":

    #for debug   
    datasets = get_train_dataloader(batch_size=16, shuffle=True, num_workers=0)
    for batch_idx, (ts_imgs, ts_masks, ts_label) in enumerate(datasets):
        print(ts_imgs.shape)
        print(ts_masks.shape)
        print(ts_label.shape)
        break
