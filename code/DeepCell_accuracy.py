import cv2
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.models import  load_model
from skimage.color import rgb2gray
from skimage import exposure
from tqdm import tqdm

#custom scripts
from segment_img import img_crop,img_reconstruct 
from features_post_process import post_proccess_mask
from seg_accuracy import seg_accuracy


save_metrics = False

fgbg = cv2.imread('/home/christos/Desktop/Thesis_test_images/test2Ec_frame_47/DeepCell/pred_inner.jpg',0)
fgbg = np.where(fgbg>150,1,0)

edge = cv2.imread('/home/christos/Desktop/Thesis_test_images/test2Ec_frame_47/DeepCell/pred_edge.jpg',0)
edge = np.where(edge>150,1,0)

res = post_proccess_mask(fgbg,edge,print_result=False)

#Load ground truth

true_fgbg = cv2.imread('/home/christos/Desktop/Thesis_test_images/test2Ec_frame_47/frame_47_Raw_Mask/test2Ec_f_47_inner.jpg',0)
true_fgbg = np.where(true_fgbg>150,1,0)

true_edge = cv2.imread('/home/christos/Desktop/Thesis_test_images/test2Ec_frame_47/frame_47_Raw_Mask/test2Ec_f_47_edge.jpg',0)
true_edge = np.where(true_edge>150,1,0)

true_res = post_proccess_mask(true_fgbg,true_edge,print_result=False)


prec,rec,cca = seg_accuracy(true_res,res)

import pickle
if save_metrics == True:
        
    with open('rec_DeepCell.txt', 'wb') as fp:
        pickle.dump(rec, fp)
        
    with open('prec_DeepCell.txt', 'wb') as fp:
        pickle.dump(prec, fp)
        
    with open('cca_DeepCell.txt', 'wb') as fp:
        pickle.dump(cca, fp)