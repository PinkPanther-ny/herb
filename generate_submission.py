import csv

import torch
from tqdm import tqdm

from src.models import ModelSelector
from src.preprocess import Preprocessor
from src.settings import configs

print("\n==================== Start generating submission ====================\n")
# Generate submission
header = ['Id', 'Predicted']

# Initialize model
model = ModelSelector(configs).get_model()
test_loader = Preprocessor().get_submission_test_loader()
print("\n==================== Dataset loaded successfully ====================\n")
print(test_loader.dataset)
print("\n==================== =========================== ====================\n")

# Get all filenames
all_ids = [int(i[0].split('-')[-1].split('.')[0]) for i in test_loader.dataset.imgs]
all_labels = []

if configs._LOAD_SUCCESS:
    model.eval()
    # No need to calculate gradients
    with torch.no_grad():
        i = 0
        for data in tqdm(iterable=test_loader, desc='Evaluating submission test set', ncols=160, unit='batches', bar_format='{l_bar}{bar:60}{r_bar}{bar:-60b}'):
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

    with open(configs.SUBMISSION_FN, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        for i in tqdm(range(len(all_labels)), "Writing answers", ncols=160, bar_format='{l_bar}{bar:60}{r_bar}{bar:-60b}'):
            # write the data
            writer.writerow([all_ids[i], all_labels[i]])
else:
    print("Fatal! Length not equal!")
