# -*- coding: utf-8 -*-
"""
Created on 27/04/2021
@author: Jason.Fang (fangjiansheng@cvte.com)
"""

import glob
import os
import SimpleITK as sitk
import numpy as np
import lidcXmlHelper as xmlHelper
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

def test():
    
    ann = pl.query(pl.Annotation).first()
    ann.print_formatted_feature_table()
    vol = ann.scan.to_volume()
    con = ann.contours[3]

    k = con.image_k_position
    ii,jj = ann.contours[3].to_matrix(include_k=False).T

    plt.imshow(vol[:,:,k], cmap=plt.cm.gray)
    plt.plot(jj, ii, '-r', lw=1, label="Nodule Boundary")
    plt.legend()
    #plt.show()
    plt.savefig('/data/pycode/VR3D/imgs/test.png')
    
def main():
    test()

if __name__ == '__main__':
    main()