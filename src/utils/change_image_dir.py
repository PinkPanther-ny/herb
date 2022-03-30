# This should only run one time to transform kaggle competition's data folder 
# structure to the type that can be accepted by ImageFolder

import os
from ..settings import configs

directory = "./train_images/"

if not os.path.exists(configs._DATA_DIR):
    os.makedirs(configs._DATA_DIR)
    
for root, subdirectories, files in os.walk(directory):
    for subdirectory in subdirectories:
        # print(os.path.join(root, subdirectory))
        pass
    for file in files:
        class_name = file.split("__")[0]
        new_dir = configs._DATA_DIR + class_name
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            
        os.rename(os.path.join(root, file), new_dir + "/" + file)
        print(os.path.join(root, file))
        pass
