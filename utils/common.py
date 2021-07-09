# encoding: utf-8
"""
Training implementation for CIFAR10 dataset  
Author: Jason.Fang
Update time: 08/07/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd

def count_bytes(file_size):
    '''
    Count the number of parameters in model
    '''
    #param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    def strofsize(integer, remainder, level):
        if integer >= 1024:
            remainder = integer % 1024
            integer //= 1024
            level += 1
            return strofsize(integer, remainder, level)
        else:
            return integer, remainder, level

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    integer, remainder, level = strofsize(int(file_size), 0, 0)
    if level+1 > len(units):
        level = -1
    return ( '{}.{:>03d} {}'.format(integer, remainder, units[level]) )

    