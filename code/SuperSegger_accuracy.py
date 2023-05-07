#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:08:40 2021

@author: christos
"""

import cv2 as cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import numpy as np
from scipy import ndimage
#custom scripts
from segment_img import img_crop,img_reconstruct 
from features_post_process import post_proccess_mask
from seg_accuracy import seg_accuracy

save_metrics=True

pred = cv2.imread('/home/cp/Documents/Thesis/Thesis_test_images/test2Ec_frame_47/superSegger/pred.jpg',0)
pred = np.where(pred>150,1,0)
plt.figure(figsize=(10,20),dpi=90)
plt.imshow(pred,cmap='gray')
plt.show()


pred_label=label(pred,connectivity=2)
    
regions=regionprops(pred_label)

for region in regions:
    if region.eccentricity < 0.4:
        pred[region.label==pred_label]=0
    if region.area < 50:
        pred[region.label==pred_label]=0

pred_label=label(pred,connectivity=2)

result=ndimage.median_filter(pred_label,size=3)
# result = bac_label

# if print_result==True:
# result[result==0]=-1
masked_array = np.ma.masked_where(result == 0, result)

plt.figure(num=None, figsize=(20, 10), dpi=90)  
cmap = plt.cm.prism
cmap.set_bad(color='black')

plt.imshow(masked_array,cmap=cmap)
#plt.savefig('sal_15_1_f_078_ad_eq.png', bbox_inches='tight')
plt.show()

#Load ground truth

true_fgbg = cv2.imread('/home/cp/Documents/Thesis/Thesis_test_images/test2Ec_frame_47/frame_47_Raw_Mask/test2Ec_f_47_inner.jpg',0)
true_fgbg = np.where(true_fgbg>150,1,0)

true_edge = cv2.imread('/home/cp/Documents/Thesis/Thesis_test_images/test2Ec_frame_47/frame_47_Raw_Mask/test2Ec_f_47_edge.jpg',0)
true_edge = np.where(true_edge>150,1,0)

true_res = post_proccess_mask(true_fgbg,true_edge,print_result=True)


prec,rec,cca,jidx = seg_accuracy(true_res,result)

import pickle
if save_metrics == True:
        
    with open('rec_SuperSegger.txt', 'wb') as fp:
        pickle.dump(rec, fp)
        
    with open('prec_SuperSegger.txt', 'wb') as fp:
        pickle.dump(prec, fp)
        
    with open('cca_SuperSegger.txt', 'wb') as fp:
        pickle.dump(cca, fp)
        
    with open('jidx_SuperSegger.txt', 'wb') as fp:
        pickle.dump(jidx, fp)