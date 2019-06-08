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
import cv2
from matplotlib import pyplot as plt


import cv2
import matplotlib.pyplot as plt
import numpy as np

def match_detector(image1,image2,detector):
    
    kp1, des1 = detector.detectAndCompute(image1,None)
    kp2, des2 = detector.detectAndCompute(image2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    good_without_list = []

    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            good_without_list.append(m)
    imageWithMatches = cv2.drawMatchesKnn(image2,kp2,image1,kp1,good,None,flags=2)

    match = np.asarray(good)
    if len(match[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in match[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in match[:,0] ]).reshape(-1,1,2)

    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(H)
    dst = cv2.warpPerspective(image1,H,(image2.shape[1] + image1.shape[1], image2.shape[0]))
    
    dst[0:image1.shape[0], 0:image1.shape[1]] = image2	
    return imageWithMatches,dst

def show_and_save(imageWithMatches,dst):
	cv2.imwrite("matches.jpg", imageWithMatches)
	plt.imshow(np.flip(imageWithMatches, axis=2))
	plt.show()
	
	cv2.imwrite("output.jpg",dst)
	plt.imshow(np.flip(dst, axis=2))
	plt.show()
	
def main():
    
    imageWithMatches,dst = match_detector("foto1A.jpg", "foto1B.jpg", cv2.xfeatures2d.SIFT_create())
    show_and_save(imageWithMatches, dst)
    
if __name__ == "__main__":
    main()
