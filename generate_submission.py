import math
import random
from matplotlib import pyplot as plt
import torch
from src.models import ModelSelector
from src.settings import configs
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.distributed as dist
from src.preprocess import Preprocessor

import csv
import random
from tqdm import tqdm, trange
import time
import os

# Transform for input images
transform = transforms.Compose([
    transforms.Resize(550),
    transforms.CenterCrop((550, 350)),
    transforms.ToTensor(),
    transforms.Normalize([0.8246, 0.7948, 0.7320], [0.1818, 0.2051, 0.2423])
    ])

# Generate submission
header = ['Id', 'Predicted']
batch_size = 128

# Initialize model
model = ModelSelector(configs).get_model()
testloader = Preprocessor(trans_test=transform).get_test_loader()
dataset = ImageFolder(root=configs._SUBMISSION_DATA_DIR, transform=transform)
print("\n==================== Dataset loaded successfully ====================\n")
print(testloader.dataset)
print("\n==================== =========================== ====================\n")

testloader_tqdm = tqdm(iterable=testloader, desc='Evaluating test set',)

# Get all filenames
all_ids = [int(i[0].split('-')[-1].split('.')[0]) for i in dataset.imgs]
all_labels = []

if configs._LOAD_SUCCESS:
    model.eval()
    # No need to calculate gradients
    with torch.no_grad():
        i=0
        for data in testloader_tqdm:
            i += 1
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(configs._DEVICE))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            all_labels = all_labels + predicted.tolist()
else:
    print("Fatal! Load model failed!")
    
print(len(all_ids), len(all_labels))
if len(all_ids) == len(all_labels):
    print(f"Total {len(all_ids)} answers\n")
    
    with open('submit_55_59875.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
        
        for i in trange(len(all_ids)):
            # write the data
            writer.writerow([all_ids[i], all_labels[i]])
else:
    print("Fatal! Length not equal!")
            
    