import os
import shutil
def copy_rename(old_file_name, new_file_name):
        src_dir= os.curdir
        dst_dir= os.path.join(os.curdir , "subfolder")
        src_file = os.path.join(src_dir, old_file_name)
        shutil.copy(src_file,dst_dir)

        dst_file = os.path.join(dst_dir, old_file_name)
        new_dst_file_name = os.path.join(dst_dir, new_file_name)
        os.rename(dst_file, new_dst_file_name)

if __name__ == "__main__":
    for i in range(5946):
        num = str(i)
        while(len(num)<4):
            num="0"+num
        new_name = "n_"+num+".png"
        copy_rename("n_0000.png",new_name)
        if(i%1000==0):
            print(i)
