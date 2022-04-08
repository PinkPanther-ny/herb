import torch
import torch.optim as optim

from ..settings import configs, logger


class OptSelector:
    # lr, momentum, dampening, weight_decay, nesterov
    sgd_param = (optim.SGD, [0.9, 0, 1e-4, True])
    # lr, lambd, alpha, t0, weight_decay
    asgd_param = (optim.ASGD, [1e-4, 0.75, 1e6, 1e-4])
    # lr
    adagrad_param = (optim.Adagrad, [])
    adam_param = (optim.Adam, [])
    adamMax_param = (optim.Adamax, [])

    basic_opts = {
        "SGD": sgd_param,
        "ASGD": asgd_param,
        "Adagrad": adagrad_param,
        "Adam": adam_param,
        "Adamax": adamMax_param

    }

    def __init__(self, params) -> None:
        self.params = params
        pass

    def get_optim(self):
        opt_info = self.basic_opts[configs.OPT]
        opt = opt_info[0](self.params, configs.LEARNING_RATE, *opt_info[1])
        if configs._LOCAL_RANK == 0:
            logger.info(f"Optimizer prototype [ {configs.OPT} ] loaded!")
            
        if configs.LOAD_SPECIFIC_MODEL and configs._LOAD_SUCCESS:
            checkpoint = torch.load(configs._MODEL_DIR + configs.MODEL_NAME, map_location=configs._DEVICE)
            opt.load_state_dict(checkpoint['opt_state_dict'])
            if configs._LOCAL_RANK == 0:
                logger.info(f"Optimizer {configs.OPT} loaded from checkpoint {configs.MODEL_NAME.replace('/', '')}!")
                
        return opt
