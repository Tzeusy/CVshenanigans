import os

data_source = "raw_mnist/0"

def write_labels_on_first_char(data_source=data_source):
    data_labels = "./"+data_source+"/label.txt"
    with open(data_labels,'w+') as f:
        for filename in os.listdir(data_source):
            if filename.endswith(".png"):
                #For now, this assigns it the label of the first character in the filename. Customize as needed.
                label = filename[0]
                f.write(filename+","+label+"\n")
            else:
                continue
        f.close()

if __name__ == '__main__':
    write_labels_on_first_char()
