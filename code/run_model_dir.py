#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:11:10 2020

@author: christos
"""
import cv2
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.models import  load_model
from skimage.color import rgb2gray
from skimage import exposure
from tqdm import tqdm
from glob import glob

#custom scripts
from segment_img import img_crop,img_reconstruct 
from features_post_process import post_proccess_mask
 
save=False
# Load fgbg model
model_fgbg = load_model('weights/model_fgbg_v10.h5')


# Load edge model
model_edge = load_model('weights/model_edge_v8.h5')


for im_path in np.sort(glob('/home/cp/Desktop/torun/HCl pH 2.5/*.tif')):
    # org_img=imread(im_path)
    # original=rgb2gray(org_img)
    original = cv2.imread(im_path,0)
    
    intensities=original.flatten()
    # p2, p98 = np.percentile(intensities[intensities != 0] , (2, 98)) 
    # original = exposure.rescale_intensity(original, in_range=(p2, p98))
    original = exposure.equalize_hist(original)*255;
    original = original.astype(np.uint8)
    plt.imshow(original,cmap='gray')
    plt.show()
    img_stck,y_index,x_index=img_crop(original) 
    
    
    fgbg_stck=np.zeros(img_stck.shape)
    edge_stck=np.zeros(img_stck.shape)
    
    print("Segmenting image \n")
    
    for i in tqdm(range(img_stck.shape[0])):
        #Get prediction fgbg model
        fgbg_p_tmp = model_fgbg.predict(np.reshape(img_stck[i,:,:],(1,256,256,1)), verbose=0)
        fgbg_stck[i,:,:] = fgbg_p_tmp[0,:,:,0]
        
        #Get prediction edge model
        edge_p_tmp = model_edge.predict(np.reshape(img_stck[i,:,:],(1,256,256,1)), verbose=0)
        edge_stck[i,:,:] = edge_p_tmp[0,:,:,0]
        
        
        
    
    
    fgbg_recon=img_reconstruct(fgbg_stck,x_index,y_index)
    edge_recon=img_reconstruct(edge_stck,x_index,y_index)
    
    plt.figure(figsize=(20,10))
    plt.imshow(fgbg_recon,cmap="gray")
    if save:
        cv2.imwrite("/home/christos/Desktop/unet/Results/bac_mask.png",fgbg_recon*255)
    plt.show()
    
    
    plt.figure(figsize=(20,10))
    plt.imshow(edge_recon,cmap="gray")
    if save:
        cv2.imwrite("/home/christos/Desktop/unet/Results/edge_mask.png",edge_recon*255)
    plt.show()
    
    res = post_proccess_mask(fgbg_recon,edge_recon,print_result=True)
    


#               DRAFT
'''
original_rgb=imread('/home/christos/Desktop/unet/ScottA_test.tif')
original=rgb2gray(original_rgb)

intensities=original.flatten()
p2, p98 = np.percentile(intensities[intensities != 0] , (2, 98)) 
original = exposure.rescale_intensity(original, in_range=(p2, p98))
#original = 255*original;
original = original.astype(np.uint8)


print(original.max())

imshow(original,cmap='gray')
plt.show()
print(original.max())

grayscale = rgb2gray(original)*255
imshow(grayscale,cmap="gray")
plt.show()

pred_fgbg=model_fgbg.predict(np.reshape(original,(1,256,256,1)), verbose=1)
imshow(pred_fgbg[0,:,:,0])
plt.show()

pred_edge=model_edge.predict(np.reshape(original,(1,256,256,1)), verbose=1)
imshow(pred_edge[0,:,:,0])
plt.show()

imshow(np.clip((pred_fgbg[0,:,:,0]-pred_edge[0,:,:,0]),0,1))
plt.show()
'''