#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:07:41 2020

@author: christos
"""
import numpy as np



def img_crop(org_img,w_height=256,w_width=256,slide_pix=6):
#img: The image inserted as an array, this script is buildto work with grayscale images
#w_height,w_width: The shape of the window
#slid_pix: number of overlapping pixels
#RETURNS: the stacked images as an array and the list with the coordinates of the cropped images
    
    # find how many windows can be aligned on width
    count_w=0
    w_pos=[]
    while(True):
        if count_w+w_width>=org_img.shape[1]:
            w_pos.append([org_img.shape[1]-w_width,org_img.shape[1]])     
            break
        else:
            w_pos.append([count_w,count_w+w_width]) 
            count_w=count_w+w_width-slide_pix
        
    
    #find how many windows can be aligned on height   
    count_h=0
    h_pos=[]
    while(True):
        
        if count_h+w_height>=org_img.shape[0]:
            h_pos.append([org_img.shape[0]-w_height,org_img.shape[0]])     
            break
        else:
            h_pos.append([count_h,count_h+w_height]) 
            count_h=count_h+w_height-slide_pix
        
    
    img_stck=np.zeros([len(w_pos)*len(h_pos),w_height,w_width])
    
    count=0

    for i in range(len(h_pos)):
        for j in range(len(w_pos)):
            img_stck[count,:,:]= org_img[ h_pos[i][0]:h_pos[i][1] , w_pos[j][0]:w_pos[j][1] ]
            count +=1
    
    return img_stck,w_pos,h_pos

def img_reconstruct(img_stck,x_index,y_index):
    
    n_x=len(x_index)
    n_y=len(y_index)
    
    img_recon=np.zeros([x_index[-1][1],y_index[-1][1]])
    
    count=0
    for i in range(n_x):
        for j in range(n_y):
            
            img_recon[x_index[i][0]:x_index[i][1],y_index[j][0]:y_index[j][1]]=img_stck[count,:,:]
            count+=1

    return img_recon