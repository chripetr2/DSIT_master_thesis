#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:37:54 2019

@author: christos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from skimage.measure import label, regionprops
from scipy import ndimage
from skimage.morphology import skeletonize, thin
from sklearn.cluster import DBSCAN

import cv2
       
def post_proccess_mask(bac_mask,edge_mask,print_result=False,compute_colonies=False,print_col_results=False):
    
    bac_mask=np.where((bac_mask)>0.4,1,0)
    edge_mask=np.where((edge_mask)>0.4,1,0)
    
    # edge_mask = skeletonize(edge_mask)
    # bac_mask=ndimage.median_filter(bac_mask,size=10)
    # edge_mask = thin(edge_mask)
    bac_mask=bac_mask-edge_mask
    bac_mask=np.where(bac_mask<0,0,bac_mask)
    # cv2.imwrite('/home/christos/Storage.HDD/Results_tmp/test/fgbg_{}.jpg'.format(f),bac_mask*255)
    # plt.figure(figsize=(20,10))
    # plt.imshow(bac_mask)
    # plt.show()    

    bac_label=label(bac_mask,connectivity=2)
    
    regions=regionprops(bac_label)
    
    for region in regions:
        if region.eccentricity < 0.4:
            bac_mask[region.label==bac_label]=0
        if region.area < 30:
            bac_mask[region.label==bac_label]=0
    
    bac_label=label(bac_mask,connectivity=2)
    
    bac_tmp_pad=np.pad(bac_label,((1,1),(1,1)),'constant')
    
    x,y=edge_mask.shape
    # 
    #apply knn 3x3
    for i in range(1,x):
        for j in range(1,y):
            if edge_mask[i,j]==1:
                lab_val,lab_c=np.unique(bac_tmp_pad[i-1:i+2,j-1:j+2],return_counts=True)
                
                if len(lab_val)!=1:
                    bac_label[i,j]=lab_val[np.argmax(lab_c[1:])+1]
                
                
    
    # result=ndimage.median_filter(bac_label,size=3)
    result = bac_label
    
    if print_result==True:
        # result[result==0]=-1
        masked_array = np.ma.masked_where(result == 0, result)
        
        figure(num=None, figsize=(20, 10), dpi=90)  
        cmap = plt.cm.prism
        cmap.set_bad(color='black')
        
        plt.imshow(masked_array,cmap=cmap)
        #plt.savefig('sal_15_1_f_078_ad_eq.png', bbox_inches='tight')
        plt.show()
    
    if compute_colonies==True:
        max_n_cells=result.max()
        center_cell=np.zeros((max_n_cells,2),dtype=float)
        for i in range(max_n_cells):
            xx,yy = np.where(result==i+1)
            center_cell[i] = [np.median(yy),np.median(xx)]
        
        colonies_l_mask = np.zeros((result.shape[0],result.shape[1]),dtype='int')
        
        if len(center_cell)<=0:pass
        else:
            clustering = DBSCAN(eps=38, min_samples=1,n_jobs=-1).fit(center_cell)
            core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
            core_samples_mask[clustering.core_sample_indices_] = True
            labels=clustering.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            
            
            for i in range(max_n_cells):
                xx,yy = np.where(result==i+1)
                colonies_l_mask[xx,yy]=labels[i]+1
            
                
        if print_col_results==True:    
            masked_col_array = np.ma.masked_where(colonies_l_mask == 0, colonies_l_mask)
                
            plt.figure(num=None)  
            cmap = plt.cm.prism
            cmap.set_bad(color='black')
            
            plt.imshow(masked_col_array,cmap=cmap)
            #plt.savefig('sal_15_1_f_078_ad_eq.png', bbox_inches='tight')
            plt.show()
            
            # Black removed and is used for noise instead.
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
            
                class_member_mask = (labels == k)
            
                xy = center_cell[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=4)
            
                xy = center_cell[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=2)
            
            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.gca().invert_yaxis()
            plt.show()
        
        return result,colonies_l_mask
    else:
        
        return result


# def colonies


