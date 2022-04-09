import math
import os
import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import Subset 
from ..settings import configs, logger


class DatasetFromSubset(Dataset):
    # https://stackoverflow.com/questions/51782021/how-to-use-different-data-augmentation-for-subsets-in-pytorch
    def __init__(self, subset:Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class Preprocessor:
    _transform_train = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation([-45, 45]),
        # transforms.AutoAugment(policy=AutoAugmentPolicy.SVHN),
        transforms.ToTensor(),
        transforms.Normalize([0.8391, 0.8141, 0.7589], [0.1644, 0.1835, 0.2162])
    ])

    _transform_test = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.8391, 0.8141, 0.7589], [0.1644, 0.1835, 0.2162])
    ])

    def __init__(self,
                 trans_train=None,
                 trans_test=None) -> None:

        self.trans_train = self._transform_train if trans_train is None else trans_train
        self.trans_test = self._transform_test if trans_test is None else trans_test

        self.loader = None
        self.test_loader = None

    def get_loader(self) -> Tuple[DataLoader, DataLoader]:

        if self.loader is not None:
            return self.loader

        data_dir = configs._DATA_DIR
        batch_size = configs.BATCH_SIZE
        n_workers = configs.NUM_WORKERS

        # ImageFolder
        dataset = ImageFolder(root=data_dir)
        logger.info(f"Dataset contains {len(dataset.classes)} classes")
        test_n_points = int(len(dataset) * configs.TEST_ON_N_PERCENT_DATA)
        train_subset, test_subset = random_split(dataset,
                                           [len(dataset) - test_n_points, test_n_points],
                                           generator=torch.Generator().manual_seed(1)
                                           )
        # Assign different transform for train/test dataset
        train_set = DatasetFromSubset(
            train_subset, transform=self.trans_train
        )
        
        test_set = DatasetFromSubset(
            test_subset, transform=self.trans_test
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
                                 shuffle=False, num_workers=n_workers)

        self.loader = (train_loader, test_loader)
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
        if classes is None:
            classes = range(len(next(os.walk(configs._DATA_DIR, topdown=True))[1]))
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
                plt.title(classes[loader.dataset[index][1]])

        fig.show()
