import math
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from ..settings import configs
from random import randrange
import os
from os import walk

class Timer:
    def __init__(self):
        self.ini = time.time()
        self.last = 0
        self.curr = 0
        
    def timeit(self)->float:
        if self.last == 0 and self.curr == 0:
            self.last = time.time()
            self.curr = time.time()
            return 0, 0
        else:
            self.last = self.curr
            self.curr = time.time()
            return time.strftime("%H:%M:%S",time.gmtime(round(self.curr - self.last, 2))), time.strftime("%H:%M:%S",time.gmtime(round(self.curr - self.ini, 2)))


def eval_total(model, testloader, timer, epoch=-1):
    # Only neccessary to evaluate model on one gpu
    if configs._LOCAL_RANK != 0:
        return
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the
    # gradients for our outputs
    with torch.no_grad():
        i=0
        pbar = tqdm(testloader, desc="Evaluating model")
        for data in pbar:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(configs._DEVICE))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i+=1
            pbar.set_postfix({'accuracy': f"{round(100 * correct / float(total), 4)}%"})
            
    print(f"{'''''' if epoch==-1 else '''Epoch ''' + str(epoch) + ''': '''}Accuracy of the network on the {total} test images: {100 * correct / float(total)} %")
    model.train()
    
    if configs.DDP_ON:
        torch.save(model.module.state_dict(), configs._MODEL_DIR + f"{100 * correct / total}".replace('.', '_') + '.pth')
    else:
        torch.save(model.state_dict(), configs._MODEL_DIR + f"{100 * correct / total}".replace('.', '_') + '.pth')


def find_best_n_model(local_rank, n=5, rand=False):
    files = next(walk(configs._MODEL_DIR), (None, None, []))[2]
    if len(files) == 0:
        return ''
    acc = sorted([float(i.split('.')[0].replace('_', '.')) for i in files], reverse=True)
    best_acc = acc[:n]
    
    for i in acc[n:]:
        try:
            os.remove(configs._MODEL_DIR + "/" + str(i).replace('.', '_') + ".pth")
        except:
            continue
            
        
    model_name = str(best_acc[randrange(n) if (rand and len(acc[:n]) == n) else 0]).replace('.', '_') + ".pth"
    if local_rank == 0:
        print(f"Loading one of the top {n} best model: {model_name}")
    return "/" + model_name


def remove_bad_models(n=5):
    files = next(walk(configs._MODEL_DIR), (None, None, []))[2]
    if len(files) == 0:
        return
    acc = sorted([float(i.split('.')[0].replace('_', '.')) for i in files], reverse=True)
    for i in acc[n:]:
        try:
            os.remove(configs._MODEL_DIR + "/" + str(i).replace('.', '_') + ".pth")
        except:
            continue

def set_random_seeds(seed=0, deterministic = True):
    """
    Set random seeds.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
        CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
        to True and `torch.backends.cudnn.benchmark` to False.
    Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


from collections import defaultdict
def eval_submission(model, testloader):
    # Only neccessary to evaluate model on one gpu
    if configs._LOCAL_RANK != 0:
        return
    model.eval()
    dict1 = defaultdict(str)
    # since we're not training, we don't need to calculate the
    # gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            # # calculate outputs by running images through the network
            # outputs = model(images.to(configs._DEVICE))
            # # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(outputs.cpu().data, 1)
            # # if predicted[0]!=11548 and predicted[0]!=3231:
            # #     print(predicted)
            print(str(images.shape))
            # if dict1[str(images.shape)]==0:
            #     dict1[str(images.shape)] = dict1[str(images.shape)] + 1
            # else:
            #     dict1[str(images.shape)] = dict1[str(images.shape)] + 1
            # if i==100:
            #     break
            # if configs.LOG_EVAL:
            #     print(images.shape)
            #     print(f"Eval: {i}/{len(testloader)}")
    print(dict1)
    model.train()

def visualize_loader(loader, n=9, train=True, rand=False, classes=None, show_classes=False, size_mul=1.0)->None:
    if classes is not None:
        configs._CLASSES = classes
        
    wid = int(math.floor(math.sqrt(n)))
    if wid * wid < n:
        wid += 1
    fig = plt.figure(figsize=(2 * wid * size_mul, 2 * wid * size_mul))

    for i in range(n):
        if rand:
            index = random.randint(0, len(loader.dataset) - 1)
        else:
            index = i
        # Add subplot to corresponding position
        fig.add_subplot(wid, wid, i + 1)
        plt.imshow((np.transpose(loader.dataset[index][0].numpy(), (1, 2, 0))))
        plt.axis('off')
        if show_classes:
            print(index)
            plt.title(configs._CLASSES[loader.dataset[index][1]])

    fig.show()
