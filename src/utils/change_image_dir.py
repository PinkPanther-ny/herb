import os

directory = "./train_images/"

if not os.path.exists("./data/"):
    os.makedirs("./data/")
    
for root, subdirectories, files in os.walk(directory):
    for subdirectory in subdirectories:
        # print(os.path.join(root, subdirectory))
        pass
    for file in files:
        class_name = file.split("__")[0]
        new_dir = "./data/" + class_name
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            
        os.rename(os.path.join(root, file), new_dir + "/" + file)
        print(os.path.join(root, file))
        pass
