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

def match_detector(name1,name2,detector):
    
    image1 = cv2.imread(name1+".jpg",cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(name2+".jpg",cv2.IMREAD_GRAYSCALE)
    
    kp1, des1 = detector.detectAndCompute(image1,None)
    kp2, des2 = detector.detectAndCompute(image2,None)

    bf      = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    good_matches = []

    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
    
    imageWithMatches = cv2.drawMatchesKnn(image2,kp2,image1,kp1,good_matches,None,flags=2)
    
    good_matches = np.asarray(good_matches)
    if len(good_matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in good_matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in good_matches[:,0] ]).reshape(-1,1,2)

    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print("Homography:")
    print(H)
    dst = cv2.warpPerspective(image1,H,(image2.shape[1] + image1.shape[1], image2.shape[0]))
    
    return imageWithMatches,dst

def show_and_save(name, nameOfDetector, imageWithMatches, dst):    
	cv2.imwrite(name+nameOfDetector+"_with_matches.jpg", imageWithMatches)	
	cv2.imwrite(name+nameOfDetector+"Perspective.jpg",dst)
	
    
def main():
    
    # Two features detectors: SIFT and SURF.
    detectors = [[cv2.xfeatures2d.SIFT_create(),"SIFT"], [cv2.xfeatures2d.SURF_create(),"SURF"]]
    # Four par of images...
    names    = [
            ["foto1A", "foto1B"],
            ["foto2A", "foto2B"],
            ["foto3A", "foto3B"],
            ["foto4A", "foto4B"]
            ]
    
    for detector in detectors:
        for name in names:
            imageWithMatches,dst = match_detector(name[0], name[1], detector[0])
            show_and_save(name[0],detector[1],imageWithMatches, dst)
    
if __name__ == "__main__":
    main()
