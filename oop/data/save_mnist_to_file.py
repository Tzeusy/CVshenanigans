import os
from pathlib import Path
from cv2 import imwrite
from mnist import read

mnist_source = Path("mnist")
mnist_train = Path(mnist_source, "train")
mnist_test = Path(mnist_source, "test")

def save_mnist(dataset, dst):    
    label_file = open(Path(dst, "label.txt"), "w")
    generator = read(dataset=dataset, path=mnist_source)

    count = [0 for i in range(10)]
    for label, image in generator:
        filename = "{}_{:04d}.png".format(label, count[label])
        
        label_file.write("{},{}\n".format(filename, label))
        imwrite(str(Path(dst, filename)), image)
        
        count[label]+= 1

    label_file.close()

os.makedirs(mnist_train, exist_ok=True)
os.makedirs(mnist_test, exist_ok=True)

save_mnist(dataset="training", dst=mnist_train)
save_mnist(dataset="testing", dst=mnist_test)
