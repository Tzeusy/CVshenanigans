import os
import mnist
from cv2 import imwrite

src = "raw_mnist"

def initialize_folders():
    for i in range(10):
        folder_name = os.path.join(src, str(i))
        os.makedirs(folder_name, exist_ok=True)

if __name__ == "__main__":
    initialize_folders()
    gen = mnist.read()
    counter = 0
    while True:
        label, img = next(gen)
        dir_name = os.path.join(src, str(label))
        filename = os.path.join(src, str(label), "{}_{:04d}.png".format(label, len(os.listdir(dir_name))))
        imwrite(filename, img)

        counter+=1
        if counter%1000 == 0:
            print(counter)
