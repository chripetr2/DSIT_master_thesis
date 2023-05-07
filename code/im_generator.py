#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 00:51:44 2020

@author: christos
"""

import glob
from natsort import natsorted
import Augmentor
import numpy as np
import cv2

imgs_mask_path = '/home/christos/Desktop/unet/data/train_v3'
# Reading and sorting the image paths from the directories
ground_truth_images = natsorted(glob.glob(imgs_mask_path +"/phase/*"))
edge_mask_images = natsorted(glob.glob(imgs_mask_path +"/edge/*.tif"))
inner_mask_images = natsorted(glob.glob(imgs_mask_path +"/inner/*.tif"))



def bac_generator(mode='inner',batch_size=128):
    if mode=='edge':
        collacted_images_and_masks = list(zip(ground_truth_images, 
                                         edge_mask_images 
                                         # ,inner_mask_images
                                         ))
        
        
    elif mode=='inner':
        
        collacted_images_and_masks = list(zip(ground_truth_images
                                         # ,edge_mask_images 
                                          ,inner_mask_images
                                         ))
        
    else:print('Something wrong with generator')

    images = [[np.asarray(cv2.imread(y,0)) for y in x] for x in collacted_images_and_masks]
    
    p=Augmentor.DataPipeline(images)

    p.zoom_random(probability=0.9, percentage_area=0.9,randomise_percentage_area=True)
    p.crop_by_size(probability=1, width=256, height=256,centre=False)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.6)
    p.flip_top_bottom(probability=0.5)
    p.random_brightness(probability=0.6, min_factor=0.7, max_factor=1.3)
    #p.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=5)
    
    generator = p.generator(batch_size)
    
    return generator


# num2gen = 20
# generator=bac_generator(mode='edge',batch_size=num2gen)
# gen_images=next(generator)
# # #number of images to be generated
# # num_im=50
# # gen_images=p.sample(num_im)
# # # 
# import random
# import matplotlib.pyplot as plt
# for h in range(num2gen):
    
#     r_index = random.randint(0, len(gen_images)-1)
#     f, axarr = plt.subplots(1, 2)
#     axarr[0].imshow(gen_images[h][0], cmap="gray")
#     axarr[1].imshow(gen_images[h][1], cmap="gray")
#     # axarr[2].imshow(gen_images[h][2], cmap="gray")
#     f.set_figheight(15)
#     f.set_figwidth(15)
#     plt.show()
# #     #f.savefig(("train_l"+str(h)+"_augm.png"), bbox_inches='tight')  