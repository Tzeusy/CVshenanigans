import os
from shutil import copyfile

#Just a simple tool to merge files from multiple subfolders into the same folder
source_dir = './raw_mnist/'

def merge_files(source_directory=source_dir):
    dest_dir = source_directory+"joint/"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for files in os.listdir(source_directory):
        for images in os.listdir(source_directory+files):
            copyfile(source_directory+files+"/"+images,dest_dir+images)

if __name__=="__main__":
    merge_files()
