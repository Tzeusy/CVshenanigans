# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:43:37 2017

@author: Tze
"""
import os
import cv2

count=0
f = open('./training/label.txt','w+')
for filename in os.listdir("jointTest"):
    if filename.endswith(".png"):
        img = cv2.imread("./training/"+filename,0)
        cv2.imshow('wat',img)
        a = cv2.waitKey(0)
        count+=1
        print("filename is " + filename)
        if(a>58 or a<49):
            print("DISASTER OCCURED STOP NOW")
            break
        if(a==49):
            print('0')
            f.write('\n'+filename+','+'0')
        elif(a==50):
            print('1')
            f.write('\n'+filename+','+'1')
        elif(a==51):
            print('2')
            f.write('\n'+filename+','+'2')
        elif(a==52):
            print('3')
            f.write('\n'+filename+','+'3')
        elif(a==53):
            print('4')
            f.write('\n'+filename+','+'4')
        elif(a==54):
            print('4')
            f.write('\n'+filename+','+'4')
        elif(a==55):
            print('4')
            f.write('\n'+filename+','+'4')
        elif(a==56):
            print('4')
            f.write('\n'+filename+','+'4')
        elif(a==57):
            print('4')
            f.write('\n'+filename+','+'4')
        elif(a==58):
            print('4')
            f.write('\n'+filename+','+'4')
    else:
        continue
f.close()
