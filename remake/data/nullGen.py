from shutil import copyfile

src_path = "./raw_mnist/null/n_0000.png"

for i in range(1, 6000):
    index = str(i)
    while(len(index)<4):
        index="0"+index
    dst_path = "./raw_mnist/null/n_"+index+".png"
    copyfile(src_path, dst_path)
