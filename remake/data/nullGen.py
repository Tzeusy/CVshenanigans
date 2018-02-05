from shutil import copyfile

src_path = "./null/0_0000.png"

for i in range(5976):
    index = str(i)
    while(len(index)<4):
        index="0"+index
    dst_path = "./null/0_"+index+".png"
    copyfile(src_path, dst_path)
