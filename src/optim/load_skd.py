import torch
from torch.optim import lr_scheduler
from ..settings import configs


class SkdSelector:
    # milestones, gamma
    MultiStepLR_param = (lr_scheduler.MultiStepLR, [configs.LEARNING_RATE_DECREASE_EPOCHS, configs.LEARNING_RATE_GAMMA])

    basic_skds = {
        "MultiStepLR": MultiStepLR_param,

    }

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        pass

    def get_skd(self):
        skd_info = self.basic_skds[configs.SKD]
        skd = skd_info[0](self.optimizer, *skd_info[1])
        if configs._LOCAL_RANK == 0:
            print(f"Scheduler prototype [ {configs.SKD} ] loaded!")
            
        if configs.LOAD_SPECIFIC_MODEL and configs._LOAD_SUCCESS:
            checkpoint = torch.load(configs._MODEL_DIR + configs.MODEL_NAME, map_location=configs._DEVICE)
            skd.load_state_dict(checkpoint['skd_state_dict'])
            if configs._LOCAL_RANK == 0:
                print(f"Scheduler {configs.SKD} loaded from checkpoint {configs.MODEL_NAME.replace('/', '')}!")
                
        return skd
