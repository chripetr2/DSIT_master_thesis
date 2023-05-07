#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:11:02 2020

@author: christos
"""


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model


#custom scripts
from models import unet
from im_generator import bac_generator


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1


inputs, outputs=unet(IMG_HEIGHT,IMG_WIDTH)  
  
model_fgbg = Model(inputs=[inputs], outputs=[outputs])
model_fgbg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print(model_fgbg.summary())


model_edge = Model(inputs=[inputs], outputs=[outputs])
model_edge.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#print(model_edge.summary())  


#Start the training

EPOCHS=40000

# # Fit model for forground-background

inner_gen=bac_generator(mode='inner',batch_size=32)
# for i in range(EPOCHS):
#     train_data=next(inner_gen)
#     X_train = np.array([data[0].reshape((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)) for data in train_data])
#     Y_train = np.array([data[1].reshape((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))/255. for data in train_data])
    
#     results_fgbg = model_fgbg.train_on_batch(X_train,Y_train)
#     print('Epoch: {}/{} - loss: {} - accuracy: {}'.format(i+1,EPOCHS,results_fgbg[0],results_fgbg[1]))

# model_fgbg.save('model_fgbg_v10.h5')



# Fit model for edge fo the bacteria

edge_gen=bac_generator(mode='edge',batch_size=32)
# for i in range(EPOCHS):
#     train_data=next(edge_gen)
#     X_train = np.array([data[0].reshape((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)) for data in train_data])
#     Y_train = np.array([data[1].reshape((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))/255. for data in train_data])
    
#     results_edge = model_edge.train_on_batch(X_train,Y_train)
#     print('Epoch: {}/{} - loss: {} - accuracy: {}'.format(i+1,EPOCHS,results_edge[0],results_edge[1]))

# model_edge.save('model_edge_v10.h5')

import matplotlib.pyplot as plt 

for i,imgs in enumerate(inner_gen):
    if i ==10:break
    tmp = imgs[0][0]






    