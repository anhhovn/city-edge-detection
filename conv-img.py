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

def plot_two_images(img1: np.array, img2: np.array):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')


#display an image

img=Image.open('cat.jpg')
img=ImageOps.grayscale(img)
img = img.resize(size=(224,224))
#plot_image(img=img)    
#declare filter for convolution
#using kernel from https://setosa.io/ev/image-kernels/
#I'm using sharpen, blur and outline

sharpen = np.array([
    [0, -1 ,0],
    [-1, 5, -1],
    [0, -1, 0]
    ])

blur = np.array([
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625]
    ])

outline = np.array([
    [-1, -1 ,-1],
    [-1, 8, -1],
    [-1, -1, -1]
    ])


#caluculate the target size for convoution implementation later
def cal_target_size(img_size: int, kernel_size: int):
    num_pixels = 0
    for i in range(img_size):
        added = i+ kernel_size
        if added <= img_size:
            num_pixels += 1
    return num_pixels
    
#Implement convolution
def convolution(img: np.array, kernel: np.array) -> np.array:
    target_size = cal_target_size(img_size = img.shape[0], kernel_size=kernel.shape[0])
    k = kernel.shape[0]
    #get 2d array of zeros
    convolved_img = np.zeros(shape=(target_size,target_size))
    
    #iteration
    for i in range(target_size):
        for j in range(target_size):
            #get the current matrix
            mat = img[i:i+k, j:j+k]
            
            #apply convolution (element-wise multiplication and summation of the results)
            convolved_img[i,j] = np.sum(np.multiply(mat, kernel))
    return convolved_img

outlined_image = convolution(img=np.array(img), kernel=outline)

plot_two_images(
    img1=img, 
    img2=outlined_image
)