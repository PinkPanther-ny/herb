
import os


for root, subdirectories, files in os.walk('/datav/alvin/herb/data/'):
    for subdirectory in subdirectories:
        if int(subdirectory) >= 1500:
            os.system(f"rm -rf {os.path.join(root, subdirectory)}")
            print(os.path.join(root, subdirectory))
        pass