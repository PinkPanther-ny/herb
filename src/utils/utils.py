import csv
import math
import os
import random
import time
from random import randrange
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..settings import configs, logger


class Timer:
    def __init__(self):
        self.ini = time.time()
        self.last = 0
        self.curr = 0

    def timeit(self) -> Union[tuple[int, int], tuple[str, str]]:
        if self.last == 0 and self.curr == 0:
            self.last = time.time()
            self.curr = time.time()
            return 0, 0
        else:
            self.last = self.curr
            self.curr = time.time()
            return time.strftime("%H:%M:%S", time.gmtime(round(self.curr - self.last, 2))), \
                   time.strftime("%H:%M:%S", time.gmtime(round(self.curr - self.ini, 2)))


def eval_total(model, test_loader, epoch=-1)->None:
    # Only necessary to evaluate model on one gpu
    if configs._LOCAL_RANK != 0:
        return
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the
    # gradients for our outputs
    with torch.no_grad():

        p_bar = tqdm(test_loader, desc=f"{'Evaluating model':18s}", ncols=160, colour='green', unit='batches')
        for data in p_bar:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(configs._DEVICE))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            result:torch.Tensor = (predicted == labels)
            correct += result.sum().item()
            p_bar.set_postfix({'accuracy': f"{round(100 * correct / float(total), 4)}%"})

    logger.info(
        f"{'' if epoch == -1 else 'Epoch ' + str(epoch) + ': '}"
        f"Accuracy of the network on the {total} test images: {100 * correct / float(total)} %")
    model.train()
    return round(100 * correct / float(total), 4)

def save_checkpoint(model, accuracy, optimizer, scheduler, epoch):
    
    acc = str(accuracy)
    model_name = acc.replace('.', '_') + '.pth'
    if configs.DDP_ON:
        torch.save({
            'epoch':epoch,
            'net_state_dict': model.module.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'skd_state_dict': scheduler.state_dict(),
                    },
                   configs._MODEL_DIR + model_name)
    else:
        torch.save({
            'epoch':epoch,
            'net_state_dict': model.module.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'skd_state_dict': scheduler.state_dict(),
                    },
                   configs._MODEL_DIR + model_name)
    logger.info(f"Epoch {epoch}: Saved checkpoint to {model_name}")


def find_best_n_model(local_rank, n=5, rand=False):
    files = next(os.walk(configs._MODEL_DIR), (None, None, []))[2]
    models = []
    for i in files:
        if i.endswith('.pth'):
            models.append(i)
    
    if len(models) == 0:
        return ''
    acc = sorted([float(i.split('.')[0].replace('_', '.')) for i in models], reverse=True)
    best_acc = acc[:n]

    for i in acc[n:]:
        try:
            os.remove(configs._MODEL_DIR + "/" + str(i).replace('.', '_') + ".pth")
        except:
            continue

    model_name = str(best_acc[randrange(n) if (rand and len(acc[:n]) == n) else 0]).replace('.', '_') + ".pth"
    if local_rank == 0:
        logger.info(f"Loading one of the top {n} best model: {model_name}")
    return "/" + model_name


def remove_bad_models(n=5):
    files = next(os.walk(configs._MODEL_DIR), (None, None, []))[2]
    
    models = []
    for i in files:
        if i.endswith('.pth'):
            models.append(i)
    if len(models) == 0:
        return
    
    acc = sorted([float(i.split('.')[0].replace('_', '.')) for i in models], reverse=True)
    for i in acc[n:]:
        try:
            os.remove(configs._MODEL_DIR + "/" + str(i).replace('.', '_') + ".pth")
        except:
            continue


def set_random_seeds(seed=0, deterministic=True):
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


def gen_submission(model, test_loader):
    logger.info("==================== Start generating submission ====================\n\n")
    # Generate submission
    header = ['Id', 'Predicted']

    # Initialize model
    # model = ModelSelector(configs).get_model()
    # testloader = Preprocessor().get_submission_test_loader()
    logger.info("==================== Dataset loaded successfully ====================\n\n")
    logger.info(test_loader.dataset)
    logger.info("==================== =========================== ====================\n\n")

    # Get all filenames
    all_ids = [int(i[0].split('-')[-1].split('.')[0]) for i in test_loader.dataset.imgs]
    all_labels = []

    if configs._LOAD_SUCCESS:
        model.eval()
        # No need to calculate gradients
        with torch.no_grad():
            i = 0
            for data in tqdm(iterable=test_loader, desc='Evaluating submission test set', ncols=160, unit='batches'):
                i += 1
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images.to(configs._DEVICE))
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                all_labels = all_labels + predicted.tolist()
    else:
        logger.warning("Fatal! Load model failed!")

    if len(all_ids) == len(all_labels):
        logger.info(f"Total {len(all_labels)} answers\n")

        with open(configs.SUBMISSION_FN, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            for i in tqdm(range(len(all_labels)), "Writing answers", ncols=160):
                # write the data
                writer.writerow([all_ids[i], all_labels[i]])
    else:
        logger.warning("Fatal! Length not equal!")


def visualize_loader(loader, n=9, rand=False, classes=None, show_classes=False, size_mul=1.0) -> None:
    if classes is None:
        classes = range(len(next(os.walk(configs._DATA_DIR, topdown=True))[1]))

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
            plt.title(classes[loader.dataset[index][1]])

    fig.show()
