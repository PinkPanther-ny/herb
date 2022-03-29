import csv
import random
import os

# Generate random submission

header = ['Id', 'Predicted']
directory = "./test_images/"

all_files = []
for root, subdirectories, files in os.walk(directory):
    for subdirectory in subdirectories:
        # print(os.path.join(root, subdirectory))
        pass
    for file in files:
        all_files.append(int(file.split('-')[1].split('.')[0]))
        pass
all_files = sorted(all_files)

with open('random.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)
    
    for i in all_files:
        # write the data
        writer.writerow([i, random.randint(0, 15504)])