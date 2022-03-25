import torch
from src.preprocess import Preprocessor


def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    count_i = 0
    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        
        count_i += 1
        if count_i%20==0:
            print(fst_moment, torch.sqrt(snd_moment - fst_moment ** 2))

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std


trainloader, testloader = Preprocessor().get_loader()

mean, std = batch_mean_and_sd(trainloader)
print("mean and std: \n", mean, std)