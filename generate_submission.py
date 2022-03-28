import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import csv

from src.models import ModelSelector
from src.settings import configs
from src.preprocess import Preprocessor


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

# Get all filenames
all_ids = [int(i[0].split('-')[-1].split('.')[0]) for i in dataset.imgs]
all_labels = []

if configs._LOAD_SUCCESS:
    model.eval()
    # No need to calculate gradients
    with torch.no_grad():
        i=0
        for data in tqdm(iterable=testloader, desc='Evaluating test set'):
            i += 1
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(configs._DEVICE))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            all_labels = all_labels + predicted.tolist()
else:
    print("Fatal! Load model failed!")
    
if len(all_ids) == len(all_labels):
    print(f"Total {len(all_labels)} answers\n")
    
    with open('submit_55_59875.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
        
        for i in tqdm(range(len(all_labels)), "Writing answers"):
            # write the data
            writer.writerow([all_ids[i], all_labels[i]])
else:
    print("Fatal! Length not equal!")
            
    