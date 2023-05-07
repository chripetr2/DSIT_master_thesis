#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:19:27 2019

@author: christos
"""
import os
import Augmentor
import numpy as np
from PIL import Image
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
#%matplotlib inline

# Reading and sorting the image paths from the directories
ground_truth_images = natsorted(glob.glob("/home/christos/deepcell2/deepcell-tf/train_data/ecoli/phase/*"))
edge_mask_images = natsorted(glob.glob("/home/christos/deepcell2/deepcell-tf/train_data/ecoli/feature_0/*.tif"))
fgbg_network_mask_images = natsorted(glob.glob("/home/christos/deepcell2/deepcell-tf/train_data/ecoli/feature_1/*.tif"))

collacted_images_and_masks = list(zip(ground_truth_images, 
                                     edge_mask_images, 
                                     fgbg_network_mask_images))

images = [[np.asarray(Image.open(y)) for y in x] for x in collacted_images_and_masks]

for i in range(len(images)):
    for table in range(len(images[0])):
        if table==0:
           
            images[i][table]=images[i][table].astype("float64")
      
p = Augmentor.DataPipeline(images)

p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.6)
p.zoom_random(probability=0.5, percentage_area=0.5)
p.flip_top_bottom(probability=0.5)
#p.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=5)

#number of images to be generated
num_im=7000
gen_images=p.sample(num_im)

#converting mid-range values to 255, because after augmentor the values do not stay 255
for k in range(num_im):
    gen_images[k][1]=np.where(gen_images[k][1]>110,255,0)
    gen_images[k][2]=np.where(gen_images[k][2]>110,255,0)

    path = '/home/christos/Desktop/unet/data/train_v2/{}'.format(k+1)
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(path+"/images")
        os.mkdir(path+"/fgbg")
        os.mkdir(path+"/edge")
        
    im = Image.fromarray(np.uint8(gen_images[k][0]))
    im_fgbg = Image.fromarray(np.uint8(gen_images[k][2]))
    im_edge = Image.fromarray(np.uint8(gen_images[k][1]))
    
    im.save(path+'/images/{}.tif'.format(k+1))
    im_fgbg.save(path+'/fgbg/{}.tif'.format(k+1))
    im_edge.save(path+'/edge/{}.tif'.format(k+1))

    

import random

for h in range(5):
    
    r_index = random.randint(0, len(gen_images)-1)
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(gen_images[h][0], cmap="gray")
    axarr[1].imshow(gen_images[h][1], cmap="gray")
    axarr[2].imshow(gen_images[h][2], cmap="gray")
    f.set_figheight(15)
    f.set_figwidth(15)
    plt.show()
    #f.savefig(("train_l"+str(h)+"_augm.png"), bbox_inches='tight')  
   











