import random
import time
import numpy as np
import torch
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
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(configs._DEVICE))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i+=1
            if configs.LOG_EVAL:
                print(f"Eval: {i}/{len(testloader)}, {correct / float(total)}, {correct}/{total}")
            
    print(f"{'''''' if epoch==-1 else '''Epoch ''' + str(epoch) + ''': '''}Accuracy of the network on the {total} test images: {100 * correct / float(total)} %")
    t = timer.timeit()
    print(f"Evaluate delta time: {t[0]}, Already: {t[1]}")
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
        print(f"Loading one of the top {n} best model: {model_name}\n")
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
