import os
import re
from typing import Any, Callable, Dict, Optional
from PIL import Image
from torch.utils.data.dataset import Dataset, Subset

# from ..settings import logger


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
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class HerbDataset(Dataset):

    regex_filename: str = '\d{5}__\d{3}.jpg'
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 ):
        if not os.path.isdir(root):
            msg = f"Fatal!{root} is not a directory!"
            # logger.critical(msg)
            raise HerbDatasetIniError(msg)
        self.root = root
        classes, class_to_idx = self.find_classes(self.root)
        
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        
        self.samples = self.make_dataset(self.root, self.class_to_idx)
        self.loader = loader
        

    def find_classes(self, directory: str):
        classes = set()
        for root, _, files in os.walk(directory):
            
            for file in files:
                if re.match(self.regex_filename, file):
                    classes.add(file.split("__")[0])
                    # self.samples.append(os.path.join(root, file))
                else:
                    msg = f"Dataset contains an invalid file {os.path.join(root, file)}!"
                    # logger.critical(msg)
                    raise HerbDatasetIniError(msg)
        classes = sorted(list(classes))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx
        

    def make_dataset(self, 
        directory: str,
        class_to_idx: Dict[str, int]
        ):
        path_label_pairs = []
        
        for root, _, files in os.walk(directory):
            
            for file in files:
                if re.match(self.regex_filename, file):
                    class_name = file.split("__")[0]
                    path_label_pairs.append((os.path.join(root, file), class_to_idx[class_name]))
                else:
                    msg = f"Dataset contains an invalid file {os.path.join(root, file)}!"
                    # logger.critical(msg)
                    raise HerbDatasetIniError(msg)
        
        return sorted(path_label_pairs)

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