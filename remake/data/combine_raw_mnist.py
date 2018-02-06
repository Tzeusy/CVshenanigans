import os
import random
from shutil import copyfile

root = "raw_mnist"
joint = "joint"
joint_training = "jointTraining"
joint_test = "jointTest"

def create_folders():
    for folder in [joint, joint_training, joint_test]:
        os.makedirs(os.path.join(root, folder), exist_ok=True)

def merge_all_images():
    for folder in os.listdir(root):
        if(folder==joint or folder==joint_training or folder==joint_test):
            continue
        folder_name = os.path.join(root, folder)
        for image in os.listdir(folder_name):
            src_path = os.path.join(folder_name, image)
            dst_path = os.path.join(root, joint, image)
            copyfile(src_path, dst_path)


def train_test_split(test_size=0.2):
    src_folder = os.path.join(root, joint)
    image_files = list(filter(lambda f: f.endswith(".png"), os.listdir(src_folder)))

    random.shuffle(image_files)

    n = int(len(image_files) * test_size)
    train_files = image_files[:-n]
    test_files = image_files[-n:]

    for file in train_files:
        src_path = os.path.join(root, joint, file)
        dst_path = os.path.join(root, joint_training, file)
        copyfile(src_path, dst_path)

    for file in test_files:
        src_path = os.path.join(root, joint, file)
        dst_path = os.path.join(root, joint_test, file)
        copyfile(src_path, dst_path)

def write_label(folder_name):
    label_filename = os.path.join(folder_name, "label.txt")
    with open(label_filename, 'w+') as f:
        for filename in os.listdir(folder_name):
            if not filename.endswith(".png"):
                continue
            f.write("{},{}\n".format(filename, filename[0]))

create_folders()
merge_all_images()
train_test_split()
write_label(os.path.join(root, joint_training))
write_label(os.path.join(root, joint_test))
