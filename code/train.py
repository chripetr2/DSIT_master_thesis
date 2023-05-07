#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:19:34 2020

@author: christos
"""

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from PIL import ImageFile

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K

import tensorflow as tf

from models import unet

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

TRAIN_PATH = '/home/christos/Desktop/unet/data/train_v2/'
dir_path = ''

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

train_ids = next(os.walk(TRAIN_PATH))[1]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.uint8)
Y_fgbg_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
Y_edge_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')

sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    
    #Read image files iteratively
    path = TRAIN_PATH + id_
    img = imread(dir_path + path + '/images/' + id_ + '.tif')
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    
    #Append image to numpy array for train dataset
    X_train[n,:,:,0] = img
    
    #Read corresponding mask files iteratively
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    edge = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    
    #Looping through masks
    for mask_file in next(os.walk(path + '/fgbg/'))[2]:
        #Read individual masks
        mask_ = imread(dir_path + path + '/fgbg/' + mask_file)
        #Expand individual mask dimensions
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        #Overlay individual masks to create a final mask for corresponding image
        mask = np.maximum(mask, mask_)
    #Append mask to numpy array for train dataset
    Y_fgbg_train[n] = mask
    
    for edge_file in next(os.walk(path + '/edge/'))[2]:
        #Read individual masks
        mask_ = imread(dir_path + path + '/edge/' + edge_file)
        #Expand individual mask dimensions
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        #Overlay individual masks to create a final mask for corresponding image
        edge = np.maximum(edge, mask_)   
    #Append mask to numpy array for train dataset
    Y_edge_train[n] = edge

inputs, outputs=unet(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)  
  
model_fgbg = Model(inputs=[inputs], outputs=[outputs])
model_fgbg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model_fgbg.summary())

model_edge = Model(inputs=[inputs], outputs=[outputs])
model_edge.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#print(model_edge.summary())  

#Start the training


# # Fit model for forground-background
# earlystopper = EarlyStopping(patience=15, verbose=1)
# checkpointer = ModelCheckpoint('/home/christos/Desktop/unet/trained_models/model_fgbg_unet_checkpoint_train_v3.h5', verbose=1, save_best_only=True)
# results_fgbg = model_fgbg.fit(X_train, Y_fgbg_train, validation_split=0.1, batch_size=16, epochs=30
                               # , callbacks=[earlystopper, checkpointer]
                              # )


# # Fit model for the edges of the bacteria
earlystopper = EarlyStopping(patience=15, verbose=1)
checkpointer = ModelCheckpoint('/home/christos/Desktop/unet/trained_models/model_edge_unet_checkpoint_train_v3.h5', verbose=1, save_best_only=True)
results_edge = model_edge.fit(X_train, Y_edge_train, validation_split=0.1, batch_size=16, epochs=30, 
                    callbacks=[earlystopper, checkpointer])










    