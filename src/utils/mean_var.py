import torch
from ..preprocess import Preprocessor
import torchvision.transforms as transforms
from tqdm import tqdm

# Transform for input images
transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])


def batch_mean_and_sd(loader=None):
    """
    Calculate mean and standard deviation of dataloader.
    """
    if loader is None:
        loader = Preprocessor(trans_train=transform).get_loader()[0]
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    pbar = tqdm(loader, desc="Calculating mean and sd")
    for images, _ in pbar:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

        cur_mean = [round(i, 4) for i in fst_moment.tolist()]
        cur_sd = [round(i, 4) for i in torch.sqrt(snd_moment - fst_moment ** 2).tolist()]
        pbar.set_postfix({'mean': cur_mean, 'sd': cur_sd})

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    print("mean and std: \n", mean, std)
    return mean, std
