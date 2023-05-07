#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 15:49:47 2020

@author: christos
"""

import cv2
import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras.models import  load_model

from skimage import exposure
from tqdm import tqdm
import os 
import tifffile as tiff
import gc
 
#custom scripts
from segment_img import img_crop,img_reconstruct 
from features_post_process import post_proccess_mask



# Load fgbg model

# model_fgbg = load_model('/home/christos/Desktop/unet/trained_models/model_fgbg_unet_checkpoint_v2.h5')
# model_fgbg = load_model('model_fgbg_v8.h5')


# Load edge model
# model_edge = load_model('/home/christos/Desktop/unet/trained_models/model_edge_unet_checkpoint_v2.h5')
model_edge = load_model('model_edge_v8.h5')

exprs_path = "/home/christos/Desktop/unet/data/Microscope E.coli"
save_path ='/home/christos/Storage.HDD/Results_v8'
basca_fgbg_path = '/home/christos/Desktop/unet/data/Basca_fgbg_mask/median_fgnd/'

for exp in os.listdir(exprs_path):
    dir_path = "{}/{}".format(exprs_path,exp)
    imgs_path = os.listdir(dir_path)
    if os.path.exists('{}/{}'.format(save_path,exp)):continue
    os.mkdir('{}/{}'.format(save_path,exp))
    os.mkdir('{}/{}/bac_segmentation'.format(save_path,exp))
    os.mkdir('{}/{}/colonies'.format(save_path,exp))
    os.mkdir('{}/{}/combined_plots'.format(save_path,exp))
    os.mkdir('{}/{}/deep_masks'.format(save_path,exp))
    os.mkdir('{}/{}/deep_masks/fgbg'.format(save_path,exp))
    os.mkdir('{}/{}/deep_masks/edge'.format(save_path,exp))
    
    for num,img_name in tqdm(enumerate(np.sort(imgs_path))):
        # print("\n Segmenting image {}/{}".format(num+1,len(imgs_path)))
        
        img_path = '{}/{}'.format(dir_path,img_name)
        
        if img_path.split('.')[-1] not in ['jpg','png','tiff']:continue
        org_img=cv2.imread(img_path,0)
        
        # intensities=org_img.flatten()
        # p2, p98 = np.percentile(intensities[intensities != 0] , (1, 99)) 
        # org_img = exposure.rescale_intensity(org_img, in_range=(p2, p98))
    
        # org_img = original.astype(np.uint8)
        
        img_stck,y_index,x_index=img_crop(org_img,slide_pix=64)
        
           
        
        # fgbg_stck=np.zeros(img_stck.shape)
        edge_stck=np.zeros(img_stck.shape)
        
        
        #########
        
        for i in range(img_stck.shape[0]):
            #Get prediction fgbg model
            # fgbg_p_tmp = model_fgbg.predict(np.reshape(img_stck[i,:,:],(1,256,256,1)), verbose=0)
            # fgbg_stck[i,:,:] = fgbg_p_tmp[0,:,:,0]
            
            #Get prediction edge model
            edge_p_tmp = model_edge.predict(np.reshape(img_stck[i,:,:],(1,256,256,1)), verbose=0)
            edge_stck[i,:,:] = edge_p_tmp[0,:,:,0]
            
            
            
        # fgbg_recon=img_reconstruct(fgbg_stck,x_index,y_index)*255
        edge_recon=img_reconstruct(edge_stck,x_index,y_index)*255
        
        fgbg_recon = cv2.imread('{}median_fgnd_{}.png'.format(basca_fgbg_path,img_name.split('.')[0]),0)
        # plt.figure(figsize=(20,10), dpi=90)
        # plt.imshow(fgbg_recon,cmap="gray")
        # # # cv2.imwrite("/home/christos/Desktop/unet/Results/bac_mask.png",fgbg_recon*255)
        # plt.show()
        
        
        # plt.figure(figsize=(20,10), dpi=90)
        # plt.imshow(edge_recon,cmap="gray")
        # # # cv2.imwrite("/home/christos/Desktop/unet/Results/edge_mask.png",edge_recon*255)
        # plt.show()
        
        result_segm, colonies = post_proccess_mask(fgbg_recon,edge_recon,print_result=False,compute_colonies=True,print_col_results=False)
            
        fig = plt.figure(dpi=1000)
        plt.subplot(1,3,1)
        plt.imshow(org_img,cmap='gray')
        plt.axis('off')
        
        plt.subplot(1,3,2)
        masked_array = np.ma.masked_where(result_segm == 0, result_segm)
        cmap = plt.cm.prism
        cmap.set_bad(color='black')
        plt.imshow(masked_array,cmap=cmap)
        plt.axis('off')
        
        plt.subplot(1,3,3)
        masked_col_array = np.ma.masked_where(colonies == 0, colonies)
        cmap = plt.cm.prism
        cmap.set_bad(color='black')
        plt.imshow(masked_col_array,cmap=cmap)
        plt.axis('off')
        # fig.subplots_adjust(top=0.88)
        fig.tight_layout()
        
        
        fig.savefig('{}/{}/combined_plots/{}_{}.jpg'.format(save_path,exp,exp,num),bbox_inches='tight')
        plt.close(fig)
        np.savetxt('{}/{}/bac_segmentation/seg_{}.csv'.format(save_path,exp,num),result_segm.astype(np.int16),delimiter=',',fmt='%d')
        np.savetxt('{}/{}/colonies/col_{}.csv'.format(save_path,exp,num),colonies.astype(np.int16),delimiter=',',fmt='%d')
        cv2.imwrite('{}/{}/deep_masks/edge/edge_{}.jpg'.format(save_path,exp,num),edge_recon*255)
        cv2.imwrite('{}/{}/deep_masks/fgbg/fgbg_{}.jpg'.format(save_path,exp,num),fgbg_recon*255)
        
        # del fgbg_recon, edge_recon, colonies, result_segm, masked_array, masked_col_array, org_img, fgbg_stck, edge_stck
        gc.collect()