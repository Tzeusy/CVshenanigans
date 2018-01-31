# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:43:37 2017

@author: Tze
"""
import os
import cv2

count=0
f = open('./jointTraining/label.txt','w+')
for filename in os.listdir("jointTraining"):
    if filename.endswith(".png"):
        f.write(filename+", "+filename[0]+"\n")
    else:
        continue
f.close()
