#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:31:43 2021

@author: christos
"""
import glob
import numpy as np
import matplotlib.pyplot as plt


import pickle
def open_pickle(path_to_pickle):
    with open(path_to_pickle, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        
    return b


metrics_path = '/home/christos/Desktop/Thesis_test_images/test2Ec_frame_47/metrics/'

metric = ['rec','prec','cca']
paths = glob.glob(metrics_path+'*.txt')

t = np.linspace(0.2, 1, 15) 
for m in metric:
    soft_m_paths = sorted(glob.glob(metrics_path + m + '*.txt'))
    metrics=np.zeros((len(soft_m_paths),15))
    s_names=[]
    for i,s in enumerate(soft_m_paths):
        tmp = open_pickle(s)
        metrics[i,:]=np.asarray(tmp)
        s_names.append(s.split('/')[-1].split('.txt')[0].split('_')[1])
    
    if m=='cca':
        tlt='Cell Count Accuracy (CCA)'
    elif m=='prec':
        tlt='Precision'
    elif m=='rec':
        tlt='Recall'
        
    fig, ax = plt.subplots(figsize=(8,8))           
    ax.plot(t,metrics[0,:],t,metrics[1,:],t,metrics[2,:],t,metrics[3,:])
    ax.set_xlabel('IoU threshold')
    ax.set_ylabel('Percentage')
    ax.title.set_text(tlt)
    ax.grid()
    ax.legend(s_names)
    plt.ylim([0,1])
    plt.show()           
    