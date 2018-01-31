import os
from shutil import copyfile

for files in os.listdir():
    for images in os.listdir(files):
        copyfile(files+"/"+images,"./joint/"+images)
