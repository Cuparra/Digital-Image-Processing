#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:24:30 2019

@author: tiago
"""

# Tiago Moreira Trocoli da Cunha
# number: 226078
# python 3

import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt


#Load Images and converting them into grayscale
def load_images(image1, image2):
    
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    gray1 = cv2.imread(image1,cv2.IMREAD_GRAYSCALE)
    gray2 = cv2.imread(image2,cv2.IMREAD_GRAYSCALE)
    
    return (img1, img2, gray1, gray2)


# find the k best matches from one image to another
def find_matches(gray1, gray2, des1, des2, kp1, kp2, min_matches):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.75*m[1].distance:
            good.append(m)
            matches = np.asarray(good)
    if len(matches[:,0]) >= 4:
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("output.jpg",img3)
    
    

def join_show_images(image1, image2, H):
    dst = cv2.warpPerspective(image1,H,(image2.shape[1] + image1.shape[1], image2.shape[0]))
    plt.subplot(122)
    plt.imshow(dst)
    plt.title("Warped Image")
    plt.show()
    plt.figure()
    dst[0:image2.shape[0], 0:image2.shape[1]] = image2
    cv2.imwrite("output.jpg",dst)
    plt.imshow(dst)
    plt.show()

# SIFT algorithm
def scale_invariante_feature(image1, image2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1,None)
    kp2, des2 = sift.detectAndCompute(image2,None)
    return kp1, kp2, des1, des2

# SURF algorithm
def speeded_up_robust_features(image1, image2):
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(image1,None)
    kp2, des2 = surf.detectAndCompute(image2,None)
    return kp1, kp2, des1, des2

def main():
    
    image1, image2, gray1, gray2 = load_images("foto1A.jpg", "foto1B.jpg")
    kp1, kp2, des1, des2         = speeded_up_robust_features(image1, image2)
    find_matches(gray1, gray2, des1, des2, kp1, kp2,10)
    
if __name__ == "__main__":
    main()