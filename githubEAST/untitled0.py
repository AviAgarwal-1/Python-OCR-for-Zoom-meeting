#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:51:47 2020

@author: tekhawk
"""

import cv2

loc= r'/home/tekhawk/githubEAST/demo_images/test-600.png'

output = cv2.imread(loc)

startX = 100
startY = 970
endX   = 280
endY   = 1000
crop = [0.135, 0.95 , 0.095 , 0.84]
#        "crop"           : [0.7, 0.97 , 0.000001 ,0.2] ,
#        "main_screen"    : [0.135, 0.95 , 0.095 , 0.84]

(h, w) = output.shape[:2]

def cropper( he, wi, crop) :
    x1= float(crop[0])
    x2= float(crop[1])
    y1= float(crop[2])
    y2= float(crop[3])

    # dimesions to crop
    start_row , start_col = int(he*x1) , int(wi*y1)
    end_row , end_col = int(he*x2) , int(wi*y2)
    # crp = cropped image
    crp = output[start_row:end_row , start_col:end_col]
    cv2.imshow("output",crp)
    cv2.waitKey(2)
    # return cropped image
    return crp


image = cropper(h, w, crop)

