
import os

# Script for creating mini test dataset

origin_data_dir = '/datav/alvin/herb/data/'
mini_data_dir = '/datav/alvin/herb/mini_data/'
n_classes = 1000

for root, subdirectories, files in os.walk(origin_data_dir):
    i=0
    for subdirectory in subdirectories:
        if i < n_classes:
            os.system(f"cp -r {os.path.join(root, subdirectory)} {mini_data_dir}")
            print(os.path.join(root, subdirectory))
            i+=1
        else:
            break
