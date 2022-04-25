# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:02:37 2022
"""

import numpy as np
from PIL import Image,ImageOps
import matplotlib.pyplot as plt


#functions display images

def plot_image(img :np.array):
    plt.figure(figsize = (6,6))
    plt.imshow(img, cmap='gray')

def plot_two_imgages(img1: np.array, img2: np.array):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')


#display an image

img=Image.open('cat.jpg')
img=ImageOps.grayscale(img)
img = img.resize(size=(224,224))
plot_image(img=img)    

