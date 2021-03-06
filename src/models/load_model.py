import os
import torch
import torchvision.models as torch_model
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn

from ..models import aresnet
from ..models import cifar10_resnet
from ..settings import configs, logger
from ..utils import find_best_n_model


class ModelSelector:
    aresnet_param = (aresnet.Net, [])

    torch_resnet18_param = (torch_model.resnet18, [])
    torch_resnet34_param = (torch_model.resnet34, [])
    torch_resnet50_param = (torch_model.resnet50, [])
    torch_resnet101_param = (torch_model.resnet101, [])
    torch_resnet152_param = (torch_model.resnet152, [])
    torch_wide_resnet50_param = (torch_model.wide_resnet50_2, [])
    torch_wide_resnet101_param = (torch_model.wide_resnet101_2, [])

    c_resnet18_param = (cifar10_resnet.resnet18, [])
    c_resnet34_param = (cifar10_resnet.resnet34, [])
    c_resnet50_param = (cifar10_resnet.resnet50, [])
    c_resnet101_param = (cifar10_resnet.resnet101, [])
    c_resnet152_param = (cifar10_resnet.resnet152, [])
    c_wide_resnet50_param = (cifar10_resnet.wide_resnet50_2, [])
    c_wide_resnet101_param = (cifar10_resnet.wide_resnet101_2, [])

    basic_net = {
        "aResnet": aresnet_param,

        "resnet18": torch_resnet18_param,
        "resnet34": torch_resnet34_param,
        "resnet50": torch_resnet50_param,
        "resnet101": torch_resnet101_param,
        "resnet152": torch_resnet152_param,
        "wide_resnet50": torch_wide_resnet50_param,
        "wide_resnet101": torch_wide_resnet101_param,

        "c_resnet18": c_resnet18_param,
        "c_resnet34": c_resnet34_param,
        "c_resnet50": c_resnet50_param,
        "c_resnet101": c_resnet101_param,
        "c_resnet152": c_resnet152_param,
        "c_wide_resnet50": c_wide_resnet50_param,
        "c_wide_resnet101": c_wide_resnet101_param,

    }

    def __init__(self) -> None:
        pass

    def get_model(self):
        
        net_info = self.basic_net[configs.MODEL]
        model:nn.Module = net_info[0](*net_info[1], num_classes=len(next(os.walk(configs._DATA_DIR, topdown=True))[1]))
        if configs._LOCAL_RANK == 0:
            logger.info(f"Model prototype [ {configs.MODEL} ] loaded!")
        # Load model to gpu
        # Check if load specific model or load best model in model folder
        if configs.LOAD_SPECIFIC_MODEL:
            if configs.LOAD_BEST_MODEL:
                configs.MODEL_NAME = find_best_n_model(configs._LOCAL_RANK)
            try:
                checkpoint = torch.load(configs._MODEL_DIR + configs.MODEL_NAME, map_location=configs._DEVICE)
                model.load_state_dict(checkpoint['net_state_dict'])
                configs._CUR_EPOCHS = checkpoint['epoch'] + 1
                configs._LOAD_SUCCESS = True
                if configs._LOCAL_RANK == 0:
                    logger.info(f"Model {configs.MODEL_NAME.replace('/', '')} loaded!")
            except FileNotFoundError:
                if configs._LOCAL_RANK == 0:
                    logger.warning(f"[\"{configs.MODEL_NAME}\"] Model not found! Fall back to untrained model.\n")
                configs._LOAD_SUCCESS = False
            except IsADirectoryError:
                if configs._LOCAL_RANK == 0:
                    logger.warning(f"{configs._MODEL_DIR} is empty! Fall back to untrained model.\n")
                configs._LOAD_SUCCESS = False

        # Move loaded model with parameters to gpus
        # Then warp with DDP, reducer will be constructed too.
        model.to(configs._DEVICE)
        if configs.DDP_ON:
            model = DistributedDataParallel(model, device_ids=[configs._LOCAL_RANK], output_device=configs._LOCAL_RANK)

        return model
