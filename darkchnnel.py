#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
# import pytesseract as pt
import plotly.express as px
# import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet

from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model

# from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from tensorflow import keras
from keras.utils import img_to_array,load_img
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array


# In[2]:


# pip install pytesseract


#  we individually take each file and parse into xml.etree and find the object -> bndbox. Then we extract xmin,xmax,ymin,ymax and saved those values in the dictionary. After we convert it into a pandas data frame and save that into CSV file and save it in project folder as shown below.

# In[3]:





# dark channel prior algorithm

import cv2
import math
import numpy as np
from numpy import ndarray

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res




# In[12]:


# import cv2
# directory ="D:/Devi/Singularis Software Technologies/numberplate_detection/img"
# data=[]
# for j in os.listdir(directory):
        
#     # print(j)
#     img_path=os.path.join(directory,j)
#     # print(img_path)
#     img_array=cv2.imread(img_path)
#     # print(img_array.shape)
#     img_resize=cv2.resize(img_array,(224,224))
#     # print(img_resize)
#     img_resize=img_resize/255
#     # print(img_resize)
#     img_reshape=img_resize.reshape(224,224,3,1)
#     data.append(img_reshape)
#     # labels.append(i)


# In[13]:


#
#     #dehaze
#     I = img_arr.astype('float64')/255;
#     dark = DarkChannel(I,15);
#     A = AtmLight(I,dark);
#     te = TransmissionEstimate(I,A,15);
#     t = TransmissionRefine(img_arr,te);
#     J = Recover(I,t,A,0.1);
#     cv2.imwrite("D:/Devi/Singularis Software Technologies/numberplate_detection/dehaze/"+ind,J*255);
#
#


