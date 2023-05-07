#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:52:48 2019

@author: christos
"""

import numpy as np
import matplotlib.pyplot as plt

#The accuracy of the segmentation is calculated using the intersection over unity to seperate True positives to false positives
# The metric used is cell count accuracy (cca=TP/(N+FP))

save_fig=0


# X_true=np.load('true_set_1.npy')
# pred = np.load('set_1_bad.npy')
#X_true=X_true[15:-15,15:-15]
def seg_accuracy(X_true,pred):
    f, ax = plt.subplots(1, 2,figsize=(16,8),dpi=200)
    cmap = plt.cm.prism
    cmap.set_bad(color='black')
    ax[0].imshow(np.ma.masked_where(X_true==0,X_true),cmap=cmap)
    ax[1].imshow(np.ma.masked_where(pred==0,pred),cmap=cmap)
    ax[0].title.set_text('True labels')
    ax[1].title.set_text('Predicted labels')
    plt.show()
    
    pred = pred[0:X_true.shape[0],0:X_true.shape[1]]
    if save_fig==1:
        f.savefig('Set_1_true_pred', bbox_inches='tight')  
    
    prec=[]
    recall=[]
    CCA=[]
    Jidx = []
    
    for thresh_tmp in range(35,95,5):
    #thresh=0.7 #for TP and FP
        thresh=round(thresh_tmp*0.01,2)
        print("computing for IoU thresh:", thresh)
        TP_c=0
        FN_c=0
        
        if X_true.shape==pred.shape:
            nc=np.max(X_true) # total number of true cells in the image
            npc = np.max(pred) # total number of predicted cells in the image
            IoU_total=0
            for i in range(1,nc):
               #find the according labels in the prediction and count them pixelwise
                pred_l, counts= np.unique(pred[np.where(X_true==i)],return_counts=True)
                
                target_p_tmp=np.where(X_true==i)
                target_p=list(zip(target_p_tmp[0],target_p_tmp[1]))
                
                pred_p_tmp=np.where(pred==pred_l[np.argmax(counts)])
                pred_p=list(zip(pred_p_tmp[0],pred_p_tmp[1]))
                
                #compute intersection over union  
                inner_p=len(set(pred_p).intersection(target_p))
                outter_p=len(pred_p)+len(target_p)-inner_p
                IoU=inner_p/outter_p
                # print(i,pred_l[np.argmax(counts)],IoU)
                IoU_total=IoU_total+IoU
                
                #find TP and FP cells and compute CCA
                if IoU>=thresh:
                    TP_c +=1
                else:
                    FN_c +=1
            FP_c = npc - TP_c
            
            recall_tmp=np.clip(round(TP_c/(TP_c+FN_c),2),0,1)
            prec_tmp=np.clip(round(TP_c/(TP_c+FP_c),2),0,1)
            CCA_tmp=np.clip(round(TP_c/(nc+FP_c),2),0,1)
            Jaccard_tmp = np.clip(round(TP_c/(TP_c+FN_c+FP_c),2),0,1)
            print(TP_c,FP_c,FN_c)
                
            # print("Mean IoU:",IoU_total/(nc-1),"\n")
            print("Precision:",prec_tmp)
            print("Recall:",recall_tmp)
            print("Jaccard index:",Jaccard_tmp)
            print("CCA:",CCA_tmp,"\n")
                
            prec.append(prec_tmp)
            recall.append(recall_tmp)
            Jidx.append(Jaccard_tmp)
            CCA.append(CCA_tmp)
                
    
    t = np.linspace(0.3, 0.9, len(prec)) 
    fig, ax = plt.subplots(figsize=(8,8))           
    ax.plot(t,prec,t,recall,t,CCA,t,Jidx)
    ax.set_xlabel('IoU threshold')
    ax.set_ylabel('Percentage')
    ax.grid()
    ax.legend(['Percision','Recall','CCA','Jaccard index'])
    plt.ylim([0,1])
    plt.xlim([0.3,0.9])
    plt.show()           
    
    if save_fig==1:           
        fig.savefig('set_1_bad_stats.png', bbox_inches='tight')         
    
    return prec,recall,CCA,Jidx
           
        
    





