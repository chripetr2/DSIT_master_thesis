#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:44:53 2020

@author: christos
"""

import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

img_name = 7
imwrite=False

im = cv2.imread( '/home/christos/Desktop/basca_data/sal15_1/sal15_1 065.tif',-1)
# ,cv2.IMREAD_GRAYSCALE)
im_norm = cv2.normalize(im, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
im_norm.astype(np.uint8)

# '/home/christos/Desktop/DeepCell/DeepCell/training_data/ecoli/set2/phase.tif'
plt.figure(figsize=(20,15))
plt.imshow(im_norm,cmap='gray')
plt.show()
if imwrite==True:
    cv2.imwrite('{}.jpg'.format(img_name),im_norm)

intensities=im_norm.flatten()
p2, p98 = np.percentile(intensities[intensities != 0] , (2, 98)) 
im2 = exposure.rescale_intensity(im_norm, in_range=(p2, p98),out_range=(0,255))
plt.figure(figsize = (20,15))
plt.imshow(im2,cmap='gray')
plt.show()
if imwrite==True:
    cv2.imwrite('{}.jpg'.format(img_name+1),im2)
