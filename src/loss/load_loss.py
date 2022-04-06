import torch
import torch.nn as nn

from ..settings import configs
from ..utils import find_best_n_model


class LossSelector:
    cross_entropy_param = (nn.CrossEntropyLoss, [])

    basic_loss = {
        "CrossEntropy": cross_entropy_param,
    }

    def __init__(self) -> None:
        pass

    def get_loss(self):
        loss_info = self.basic_loss[configs.LOSS]
        loss = loss_info[0](*loss_info[1])
        if configs._LOCAL_RANK == 0:
            print(f"Loss function [ {configs.LOSS} ] loaded!")
            
    
        return loss
