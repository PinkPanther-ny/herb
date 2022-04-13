import os
import csv
import re
import pandas as pd
from typing import Any, Callable, Dict, Optional
from PIL import Image
from torch.utils.data.dataset import Dataset, Subset

from ...settings import logger, configs


class HerbDatasetIniError(Exception):
    pass


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path: str) -> Any:
    import accimage
    try:
        acc_loader = accimage.Image(path)
        # logger.info("Using accimage loader")
        return acc_loader
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        pil = pil_loader(path)
        # logger.info("Using pil loader")
        return pil


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend, set_image_backend
    set_image_backend(configs.IMAGE_BACKEND)
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        # logger.info("Using pil loader")
        return pil_loader(path)


class HerbDataset(Dataset):

    regex_image_filename: str = '\d{5}__\d{3}.jpg'
    _herb_csv_filename:str = 'herbIndex.csv'
    
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 ):
        if not os.path.isdir(root):
            msg = f"Fatal! {root} is not a directory!"
            logger.critical(msg)
            raise HerbDatasetIniError(msg)
        self.root = root
        self.transform = transform
        self.loader = loader
        
        index_file_exist = os.path.isfile(root + self._herb_csv_filename)
        
        if not index_file_exist:
            classes, class_to_idx = self.find_classes(self.root)
            self.classes = classes
            self.samples = self.make_dataset(self.root, class_to_idx)
            
            if configs._LOCAL_RANK == 0:
                idx_to_class = {y:x for x,y in class_to_idx.items()}
                self.save_index(idx_to_class)
                # Store class name as string instead of casting to int
                self.df = pd.read_csv(root + self._herb_csv_filename, dtype={1:'object'})
        else:
                # Store class name as string instead of casting to int
            self.df = pd.read_csv(root + self._herb_csv_filename, dtype={1:'object'})
            self.classes = self.df['class_names'].unique().tolist()
            self.samples = [(path, label) for path, label in zip(self.df['paths'], self.df['labels'])]
            logger.info(f"Dataset loaded from index file: {self.root + self._herb_csv_filename}")
        

    def find_classes(self, directory: str):
        classes = set()
        for root, _, files in os.walk(directory):
            
            for file in files:
                if file == self._herb_csv_filename:
                    continue
                
                if re.match(self.regex_image_filename, file):
                    classes.add(file.split("__")[0])
                else:
                    msg = f"Dataset contains an invalid file {os.path.join(root, file)}!"
                    logger.critical(msg)
                    raise HerbDatasetIniError(msg)
        classes = sorted(list(classes))
        class_to_idx = {str(cls_name): i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx
        

    def make_dataset(self, 
        directory: str,
        class_to_idx: Dict[str, int]
        ):
        path_label_pairs = []
        
        for root, _, files in os.walk(directory):
            
            for file in files:
                if file == self._herb_csv_filename:
                    continue
                
                if re.match(self.regex_image_filename, file):
                    class_name = file.split("__")[0]
                    path_label_pairs.append((os.path.join(root, file), class_to_idx[class_name]))
                else:
                    msg = f"Dataset contains an invalid file {os.path.join(root, file)}!"
                    logger.critical(msg)
                    raise HerbDatasetIniError(msg)
        
        return sorted(path_label_pairs)

    def save_index(self, idx_to_class):
        
        header = ['labels', 'class_names', 'paths']
        with open(self.root + self._herb_csv_filename, 'w') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            
            for path, label in self.samples:
                class_name = idx_to_class[label]
                writer.writerow([label, class_name, path])
        logger.info(f"Dataset index file not found, saved to {self.root + self._herb_csv_filename}")

    def __getitem__(self, index: int) -> Any:
        
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        head = self.__class__.__name__
        body = ["Number of datapoints and classes: {}, {}".format(self.__len__(), len(self.classes))]
        if self.root is not None:
            body.append("Data loaded from root: {}".format(self.root))
        if hasattr(self, "transform") and self.transform is not None:
            body += ["\nTransform: " + repr(self.transform)]
        lines = [head] + [" " * 4 + line for line in body]
        return '\n'.join(lines)


# To apply different transforms for train/test datasets after split by torch.utils.data.random_split
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