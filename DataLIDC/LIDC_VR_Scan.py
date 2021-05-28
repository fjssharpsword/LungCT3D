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
from LungCT3D.config import *
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
        vol_ids = []
        vol_pixelspacings = []
        vol_imgs = []
        vol_masks = []
        vol_labels = []
        datas = pd.read_csv(path_to_dataset_file, \
                            names =['vol_id','nod_id','slice_id','img','mask','mali_label','cancer_flag','clean_flag'], \
                            sep=',', header=None)
        datas_dicom = pd.read_csv(PATH_TO_DICOM_INFO, sep=',') #read the dicominfo
        vol_list = datas['vol_id'].unique().tolist() #first col
        for vol in vol_list:
            pixelSpacing = datas_dicom.loc[datas_dicom['patient_id'] == vol].values[0][1]
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
                    mali_label = nod_line[5]-1 #[1-5]->[0,4]
                    mali_labels.append(mali_label)
                vol_ids.append(vol)
                vol_pixelspacings.append(pixelSpacing)
                vol_imgs.append(slice_names)
                vol_masks.append(mask_names)
                assert len(set(mali_labels))== 1 #equal label for the same nodule
                vol_labels.append(list(set(mali_labels)))
        
        self.vol_ids = vol_ids
        self.vol_pixelspacings = vol_pixelspacings
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
        vol_masks = self.vol_masks[index]
        ts_imgs = torch.FloatTensor()
        ts_masks = torch.FloatTensor()
        for img, mask in zip(vol_imgs, vol_masks):
            img = np.load(img)
            mask = np.load(mask)
            ind = mask.nonzero()#img = img * mask
            x_min, x_max, y_min, y_max = ind[0].min(), ind[0].max(), ind[1].min(), ind[1].max()#print(img[ind])
            img =  img[x_min-1:x_max+1, y_min-1:y_max+1]
            img = zoom(img, (config['VOL_DIMS']/img.shape[0], config['VOL_DIMS']/img.shape[1]), order=1)
            img = torch.as_tensor(img, dtype=torch.float32)
            ts_imgs = torch.cat((ts_imgs, img.unsqueeze(0)), 0)
            mask = mask[x_min-1:x_max+1, y_min-1:y_max+1]
            mask = zoom(mask, (config['VOL_DIMS']/mask.shape[0], config['VOL_DIMS']/mask.shape[1]), order=1)
            mask = torch.as_tensor(mask, dtype=torch.float32)
            ts_masks = torch.cat((ts_masks, mask.unsqueeze(0)), 0)
        ts_imgs = zoom(ts_imgs, (config['VOL_DIMS']/(2*ts_imgs.shape[0]), 1, 1), order=1)
        ts_imgs = torch.as_tensor(ts_imgs, dtype=torch.float32)
        ts_imgs = ts_imgs.unsqueeze(0) #CxDXHxW
        ts_masks = zoom(ts_masks, (config['VOL_DIMS']/(2*ts_masks.shape[0]), 1, 1), order=1)
        ts_masks = torch.as_tensor(ts_masks, dtype=torch.float32)
        ts_masks = ts_masks.unsqueeze(0)#CxDXHxW
    
        #calculate the volume of nodule
        pixelspacing = self.vol_pixelspacings[index]
        nod_vol = pixelspacing * torch.sum(ts_masks)
        ts_nodvol = torch.as_tensor(nod_vol, dtype=torch.float32)
        #get label
        ts_label = torch.as_tensor(self.vol_labels[index][0], dtype=torch.long) 
        return ts_imgs, ts_masks, ts_label, ts_nodvol

    def __len__(self):
        return len(self.vol_ids)

"""
def collate_fn(batch):
    return tuple(zip(*batch))
 #DataLoader: collate_fn=collate_fn
"""
PATH_TO_DICOM_INFO = '/data/pycode/LungCT3D/DataLIDC/LIDC_DICOM_info.txt'
PATH_TO_TRAIN_FILE = '/data/pycode/LungCT3D/DataLIDC/LIDC_VR_Train.txt'
def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_dataset_file=PATH_TO_TRAIN_FILE)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

PATH_TO_TEST_FILE = '/data/pycode/LungCT3D/DataLIDC/LIDC_VR_Test.txt'
def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_dataset_file=PATH_TO_TEST_FILE)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

if __name__ == "__main__":

    #for debug   
    datasets = get_test_dataloader(batch_size=64, shuffle=True, num_workers=8)
    for batch_idx, (ts_imgs, ts_masks, ts_label, ts_nodvol) in enumerate(datasets):
        print(ts_imgs.shape)
        print(ts_masks.shape)
        print(ts_label.shape)
        print(ts_nodvol.shape)
        break
