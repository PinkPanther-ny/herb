import random
from typing import Tuple
import torch

import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10, ImageFolder
from ..settings.configs import configs
import math
from torchvision.transforms import AutoAugmentPolicy

class Preprocessor:
    
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=AutoAugmentPolicy.SVHN),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self,
                 trans_train=transform_train,
                 trans_test=transform_test) -> None:
        self.trans_train = trans_train
        self.trans_test = trans_test
        self.loader = None

    def get_loader(self)->Tuple[DataLoader, DataLoader]:

        if self.loader is not None:
            return self.loader

        data_dir = configs._DATA_DIR
        batch_size = configs.BATCH_SIZE
        n_workers = configs.NUM_WORKERS
        data_dir = "/datav/alvin/herb/data"
        # ImageFolder
        dataset = ImageFolder(root=data_dir, transform=self.transform_train)
        train_set, test_set = random_split(dataset, [len(dataset)-80000, 80000], generator=torch.Generator().manual_seed(1))
        if configs.DDP_ON:
            train_sampler = DistributedSampler(train_set)
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      sampler=train_sampler, num_workers=n_workers, pin_memory=True, drop_last=True)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True, num_workers=n_workers, pin_memory=True, drop_last=True)
    
        # Test with whole test set, no need for distributed sampler
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=True, num_workers=n_workers)

        # Return two iterables which contain data in blocks, block size equals to batch size
        return train_loader, test_loader

    def visualize_data(self, n=9, train=True, rand=True, classes=None, show_classes=False, size_mul=1.0)->None:
        if classes is not None:
            configs._CLASSES = classes
        loader = self.get_loader()[int(not train)]
        wid = int(math.floor(math.sqrt(n)))
        if wid * wid < n:
            wid += 1
        fig = plt.figure(figsize=(2 * wid * size_mul, 2 * wid * size_mul))

        for i in range(n):
            if rand:
                index = random.randint(0, len(loader.dataset) - 1)
            else:
                index = i
            print(index)
            # Add subplot to corresponding position
            fig.add_subplot(wid, wid, i + 1)
            plt.imshow((np.transpose(loader.dataset[index][0].numpy(), (1, 2, 0))))
            plt.axis('off')
            if show_classes:
                plt.title(configs._CLASSES[loader.dataset[index][1]])

        fig.show()
