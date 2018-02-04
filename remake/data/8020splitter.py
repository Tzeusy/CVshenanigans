# -*- coding: utf-8 -*-
import os
import cv2
import random
import labelwriter as lw

count=1

data_dir = "./raw_mnist/joint"

def split_files(data_directory = data_dir):
    training_dir = data_directory+"/training/"
    testing_dir = data_directory+"/testing/"
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)
    for filename in os.listdir(data_directory):
        if filename.endswith(".png"):
            #0 for grayscale, 1 for color, -1 for unchanged (eg. alpha if png)
            #Can also use copy (see filemerger.py), but this implementation (while a bit slower)
            #allows you to convert file being read to specified file format
            img = cv2.imread(data_directory+"/"+filename,0)
            if(random.randrange(0,100)<=79):
                cv2.imwrite(training_dir+filename,img)
            else:
                cv2.imwrite(testing_dir+filename,img)
        else:
            continue
    lw.write_labels_on_first_char(data_dir+"/training")
    lw.write_labels_on_first_char(data_dir+"/testing")

if __name__=="__main__":
    split_files()
