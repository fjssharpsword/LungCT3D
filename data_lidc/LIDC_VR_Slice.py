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
        slice_names = []
        mask_names = []
        mali_levels = []
        for file_path in path_to_dataset_file:
            with open(file_path, "r") as f:
                for line in f:
                    items = line.split(',')
                    slice_name = os.path.join(PATH_TO_IMAGES_DIR, items[0], items[3]+'.npy') 
                    slice_names.append(slice_name)
                    mask_name = os.path.join(PATH_TO_MASKS_DIR, items[0], items[4]+'.npy') 
                    mask_names.append(mask_name)
                    mali_level = int(eval(items[5])) #eval for 
                    mali_levels.append([mali_level])
                
        self.slice_names = slice_names
        self.mask_names = mask_names
        self.mali_levels = mali_levels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        slice_name = np.load(self.slice_names[index])
        mask_name = np.load(self.mask_names[index])
        mali_level = torch.as_tensor(self.mali_levels[index], dtype=torch.float32)
        return slice_name, mask_name, mali_level

    def __len__(self):
        return len(self.mali_levels)

transform_seq = transforms.Compose([
    transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE'])),
    transforms.ToTensor() #to tesnor [0,1]
])
PATH_TO_TRAIN_FILE = '/data/pycode/VR3D/datasets/LIDC_VR_Train.txt'
def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_dataset_file=[PATH_TO_TRAIN_FILE])
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

PATH_TO_TEST_FILE = '/data/pycode/VR3D/datasets/LIDC_VR_Test.txt'
def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_dataset_file=[PATH_TO_TEST_FILE])
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test


def ValidateDataset():
    datas = pd.read_csv(PATH_TO_DATASET_FILE, sep=',') 
    datas['patient_id'] = datas['patient_id'].apply(lambda x: 'LIDC-IDRI-'+'{:0>4d}'.format(x) ) #LIDC-IDRI-0000
    #datas = datas[datas.is_clean=='True'].reset_index(drop=True) #remove smaples with no nodules
    
    #visualization
    vis_0003 = datas[datas["patient_id"]=='LIDC-IDRI-0003']
    vis_0003 = vis_0003[vis_0003["slice_no"]==2]
    for line in vis_0003.values: #dataframe -> numpy
        #get slice image
        slice_path = os.path.join(PATH_TO_IMAGES_DIR, line[0], line[3]+'.npy')
        np_slice = np.load(slice_path) #[0,255]
        np_slice = ( (np_slice - np_slice.min()) * (1/(np_slice.max() - np_slice.min()) ) * 255).astype('uint8')
        img_slice = Image.fromarray(np_slice).convert('RGB')
        #get mask image
        mask_path = os.path.join(PATH_TO_MASKS_DIR, line[0], line[4]+'.npy')
        np_mask = np.load(mask_path) #{0,1}
        img_mask = Image.fromarray(np_mask.astype('uint8')*255).convert('RGB') #255-white
        L, H = img_mask.size #turn white to red
        for h in range(H):
            for l in range(L):
                dot = (l,h)
                color = img_mask.getpixel(dot)
                if color == (255, 255, 255):
                    #color = ( 0 , 255, 0) #turn to green 
                    color = (255 , 0, 0) #turn to red
                    img_mask.putpixel(dot,color)
        #overlay and show
        overlay_img = cv2.addWeighted(np.array(img_slice), 0.7, np.array(img_mask), 0.3, 0)
        #overlay_img = Image.alpha_composite(img_slice, img_mask) #mode=RGBA
        plt.imshow(overlay_img)
        plt.axis('off')
        plt.savefig(config['img_path']+line[0]+'_overlay.jpg')
        break
    """
    vis_0032 = datas[datas["patient_id"]=='LIDC-IDRI-0032']
    vis_0032 = vis_0032[vis_0032["slice_no"]==15]
    for line in vis_0032.values: #dataframe -> numpy
        #get slice image
        slice_path = os.path.join(PATH_TO_IMAGES_CLEAN_DIR, line[0], line[3]+'.npy')
        np_slice = np.load(slice_path) #[0,255]
        np_slice = ( (np_slice - np_slice.min()) * (1/(np_slice.max() - np_slice.min()) ) * 255).astype('uint8')
        img_slice = Image.fromarray(np_slice).convert('RGB')
        #get mask image
        mask_path = os.path.join(PATH_TO_MASKS_CLEAN_DIR, line[0], line[4]+'.npy')
        np_mask = np.load(mask_path) #{0,1}
        img_mask = Image.fromarray(np_mask.astype('uint8')*255).convert('RGB') #255-white
        L, H = img_mask.size #turn white to red
        for h in range(H):
            for l in range(L):
                dot = (l,h)
                color = img_mask.getpixel(dot)
                if color == (255, 255, 255):
                    #color = ( 0 , 255, 0) #turn to green 
                    color = (255 , 0, 0) #turn to red
                    img_mask.putpixel(dot,color)
        #overlay and show
        overlay_img = cv2.addWeighted(np.array(img_slice), 0.7, np.array(img_mask), 0.3, 0)
        #overlay_img = Image.alpha_composite(img_slice, img_mask) #mode=RGBA
        plt.imshow(overlay_img)
        plt.axis('off')
        plt.savefig(config['img_path']+line[0]+'_overlay.jpg')
        break
    """

def SplitDataset():
    datas = pd.read_csv(PATH_TO_DATASET_FILE, sep=',') 
    datas['patient_id'] = datas['patient_id'].apply(lambda x: 'LIDC-IDRI-'+'{:0>4d}'.format(x) ) #LIDC-IDRI-0000
    datas = datas[datas.is_clean!=True].reset_index(drop=True) #remove smaples with no nodules(6885)
    #statistics: 
    case_id = datas['patient_id'].unique().tolist()
    print('Number of Case: {}'.format(len(case_id)))
    case_id_te = random.sample(case_id, int(0.1*len(case_id))) #testset
    case_id_tr = list(set(case_id).difference(set(case_id_te))) #trainset
    datas_tr = pd.DataFrame(columns = datas.columns.values.tolist())
    for case_id in case_id_tr:
        #datas_tr = pd.concat([datas_tr,datas[datas['patient_id']==case_id]])
        datas_tr = datas_tr.append(datas[datas['patient_id']==case_id])
    print('Shape of trainset: {}'.format(datas_tr.shape))
    datas_tr.to_csv('/data/pycode/VR3D/datasets/LIDC_VR_Train.txt', index=False, header=False, sep=',')
    datas_te = pd.DataFrame(columns = datas.columns.values.tolist())
    for case_id in case_id_te:
        #datas_tr = pd.concat([datas_tr,datas[datas['patient_id']==case_id]])
        datas_te = datas_te.append(datas[datas['patient_id']==case_id])
    print('Shape of trainset: {}'.format(datas_te.shape))
    datas_te.to_csv('/data/pycode/VR3D/datasets/LIDC_VR_Test.txt', index=False, header=False, sep=',')

if __name__ == "__main__":

    #ValidateDataset()
    #SplitDataset()
    
    #for debug   
    datasets = get_test_dataloader(batch_size=10, shuffle=True, num_workers=0)
    for batch_idx, (slice_name, mask_name, mali_level) in enumerate(datasets):
        print(slice_name.shape)
        print(mask_name.shape)
        print(mali_level.shape)
        break
