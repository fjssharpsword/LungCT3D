# encoding: utf-8
"""
Training implementation for CT-3D retrieval
Author: Jason.Fang
Update time: 08/05/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from skimage.measure import label
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
#self-defined
from config import *
from util.logger import get_logger
from datasets.CT3D_dataset import get_train_dataloader, get_test_dataloader
#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
logger = get_logger(config['log_path'])

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    print('********************load data succeed!********************')


def Test():
    print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

def main():
    Train()
    Test()


if __name__ == '__main__':
    main()
