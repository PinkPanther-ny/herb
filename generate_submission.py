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

configs.reset_working_dir(__file__)
model_dir = configs._MODEL_DIR + configs.MODEL_NAME
batch_size = 512
data_dir = "./test_images/"

if configs._LOCAL_RANK == 0:
    print(f"\n==================  Loading required configurations  ==================\n")
# DDP backend initialization
if configs.DDP_ON:
    torch.cuda.set_device(configs._LOCAL_RANK)
    dist.init_process_group(backend='nccl')


# Define model, loss function and optimizer for the following training process
model = ModelSelector(configs).get_model()


testloader = DataLoader(
    ImageFolder(root=configs._SUBMISSION_DATA_DIR, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                ), 
    batch_size=batch_size, shuffle=False, num_workers=4
    )

if configs._LOAD_SUCCESS:
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the
    # gradients for our outputs
    with torch.no_grad():
        i=0
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(configs._DEVICE))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.cpu().data, 1)
            print(predicted)
            print(f"Eval: {i}/{len(testloader)}, {correct / float(total)}, {correct}/{total}")
else:
    print("Fatal! Load model failed!")