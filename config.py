import os

config = {
            'CKPT_PATH': '/data/pycode/LungCT3D/ckpt/',
            'log_path':  '/data/pycode/LungCT3D/log/',
            'img_path': '/data/pycode/LungCT3D/imgs/',
            'CUDA_VISIBLE_DEVICES': "0,1,2,3,4,5,6,7",
            'VOL_DIMS': [80, 80, 80], #D, H, W
            'MAX_EPOCHS': 200,
            'BATCH_SIZE': 80
         } 

#config for dataset
PATH_TO_DATASET_FILE = '/data/fjsdata/LIDC-IDRI/vrdata/Meta/meta_info.csv'
PATH_TO_IMAGES_DIR = '/data/fjsdata/LIDC-IDRI/vrdata/Image/'
PATH_TO_MASKS_DIR = '/data/fjsdata/LIDC-IDRI/vrdata/Mask/'
PATH_TO_IMAGES_CLEAN_DIR = '/data/fjsdata/LIDC-IDRI/vrdata/Clean/Image/'
PATH_TO_MASKS_CLEAN_DIR = '/data/fjsdata/LIDC-IDRI/vrdata/Clean/Mask/'
