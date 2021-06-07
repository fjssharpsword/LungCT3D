# -*- coding: utf-8 -*-
"""
Created on 13/05/2021
@author: Jason.Fang (fangjiansheng@cvte.com)
"""

import glob
import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import pylidc as pl
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib.widgets import Slider

#ref:
#https://github.com/jaeho3690/LIDC-IDRI-Preprocessing
#https://github.com/MIC-DKFZ/LIDC-IDRI-processing
#https://dicom.innolitics.com/ciods
"""
install pylidc: 
https://pylidc.github.io/  
https://github.com/notmatthancock/pylidc
-----------config /root/.pylidcrc-------------
[dicom]
path = /data/fjsdata/LIDC-IDRI/LIDC-IDRI-All/
warn = True
---------------------------------------------
"""
#https://pylidc.github.io/scan.html
#https://dicom.innolitics.com/ciods/ct-image/image-plane/00280030
def getPixelspacing():
    """
    scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1)
    print(scans.count())
    # => 97
    scan = scans.first()
    print(scan.patient_id, scan.pixel_spacing,scan.slice_thickness,scan.slice_spacing)
    # => LIDC-IDRI-0066, 0.63671875, 0.6, 0.5
    print(len(scan.annotations))
    """
    scans = pl.query(pl.Scan)
    df_info = pd.DataFrame(columns = ['patient_id','pixel_spacing','slice_thickness','slice_spacing'])
    for scan in scans:
        #print([scan.patient_id, scan.pixel_spacing,scan.slice_thickness,scan.slice_spacing])
        df_info = df_info.append([{'patient_id':scan.patient_id, 'pixel_spacing':scan.pixel_spacing,\
                                    'slice_thickness':scan.slice_thickness,'slice_spacing':scan.slice_spacing}], ignore_index=True)
    print('Shape of DICOM Info: {}'.format(df_info.shape))
    df_info.to_csv('/data/pycode/LungCT3D/DataLIDC/LIDC_DICOM_info.txt', index=False, sep=',')

def main():
    getPixelspacing()

if __name__ == '__main__':
    main()