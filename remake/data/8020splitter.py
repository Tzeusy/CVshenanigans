# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:29:42 2017

@author: Tze
"""

import os
import cv2
import random

count=1
for filename in os.listdir("./joint"):
    if filename.endswith(".png"):
        a = random.randrange(0,100)
        img = cv2.imread("./joint/"+filename,0)
        if(a<=79):
            cv2.imwrite('./jointTraining/'+filename,img)
        else:
            cv2.imwrite('./jointTest/'+filename,img)
        print(filename)
#        img = cv2.imread(filename,0)
#        print(img.shape)
#        crop_img = img[160:272, -485:-100] # Crop from x, y, w, h -> 100, 200, 300, 400
#        cv2.imshow('wat',crop_img)
#        a = cv2.waitKey(0)
#        print(a)
#        count+=1
        continue
    else:
        continue
