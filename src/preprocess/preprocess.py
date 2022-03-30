import math
import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

from ..settings.configs import configs


class Preprocessor:
    transform_train = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation([-45, 45]),
        # transforms.AutoAugment(policy=AutoAugmentPolicy.SVHN),
        transforms.ToTensor(),
        transforms.Normalize([0.8391, 0.8141, 0.7589], [0.1644, 0.1835, 0.2162])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.8391, 0.8141, 0.7589], [0.1644, 0.1835, 0.2162])
    ])

    def __init__(self,
                 trans_train=None,
                 trans_test=None) -> None:

        self.trans_train = self.transform_train if trans_train is None else trans_train
        self.trans_test = self.transform_test if trans_test is None else trans_test

        self.loader = None
        self.test_loader = None

    def get_loader(self) -> Tuple[DataLoader, DataLoader]:

        if self.loader is not None:
            return self.loader

        data_dir = configs._DATA_DIR
        batch_size = configs.BATCH_SIZE
        n_workers = configs.NUM_WORKERS

        # ImageFolder
        dataset = ImageFolder(root=data_dir, transform=self.trans_train)
        train_set, test_set = random_split(dataset,
                                           [len(dataset) - configs.TEST_N_DATA_POINTS, configs.TEST_N_DATA_POINTS],
                                           generator=torch.Generator().manual_seed(1)
                                           )

        if configs.DDP_ON:
            train_sampler = DistributedSampler(train_set)
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      sampler=train_sampler, num_workers=n_workers, pin_memory=False, drop_last=True)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True, num_workers=n_workers, pin_memory=False, drop_last=True)

        # Test with whole test set, no need for distributed sampler
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=True, num_workers=n_workers)

        # Return two iterables which contain data in blocks, block size equals to batch size
        return train_loader, test_loader

    def get_submission_test_loader(self) -> DataLoader:
        if self.test_loader is not None:
            return self.test_loader

        batch_size = configs.BATCH_SIZE
        n_workers = configs.NUM_WORKERS
        dataset = ImageFolder(root=configs._SUBMISSION_DATA_DIR, transform=self.trans_test)
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size, shuffle=False, num_workers=n_workers
        )

        self.test_loader = test_loader
        return self.test_loader

    def visualize_data(self, n=9, train=True, rand=True, classes=None, show_classes=False, size_mul=1.0) -> None:
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
            # Add subplot to corresponding position
            fig.add_subplot(wid, wid, i + 1)
            plt.imshow((np.transpose(loader.dataset[index][0].numpy(), (1, 2, 0))))
            plt.axis('off')
            if show_classes:
                print(index)
                plt.title(configs._CLASSES[loader.dataset[index][1]])

        fig.show()
