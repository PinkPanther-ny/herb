import os
import json

import torch


class Config:
    def __init__(self, *dict_config) -> None:
        # ==============================================
        # GLOBAL SETTINGS
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1, 2, 3, 4, 5, 6, 7]))
        
        self.DDP_ON: bool = True
        self.MIX_PRECISION: bool = True

        self.BATCH_SIZE: int = 170
        
        self.LEARNING_RATE: float = 1e-2
        self.LEARNING_RATE_DECREASE_EPOCHS = [5,10,15,20]
        self.LEARNING_RATE_GAMMA = 0.4
        
        self.TOTAL_EPOCHS: int = 5000

        self.LOAD_MODEL: bool = True
        self.MODEL_NAME: str = "53_05.pth"
        self.LOAD_BEST: bool = False
        self.LOG_EVERY_TIME:bool = False
        self.LOG_EVAL:bool = True
        self.N_LOGS_PER_EPOCH: int = 100

        self.GEN_SUBMISSION:bool = True

        # ==============================================
        # SPECIAL SETTINGS
        
        # Select in optim/load_opt and loss/load_loss
        self.MODEL = "resnet50"
        self.OPT = "Adam"
        self.LOSS = "CrossEntropy"
        
        self.EPOCHS_PER_EVAL: int = 1
        self.NUM_WORKERS: int = 12
        self.MODEL_DIR_NAME: str = "/models_v100/"
        
        # ==============================================
        # Private
        cur_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self._WORKING_DIR: str = os.path.join('/', *cur_dir.split("/")[:-2])
        self._MODEL_DIR: str = self._WORKING_DIR + self.MODEL_DIR_NAME
        self._DATA_DIR: str = self._WORKING_DIR + '/data/'
        self._SUBMISSION_DATA_DIR:str = self._WORKING_DIR + "/test_images/"
        
        self._NUM_CLASSES: int = 15505
        self._CLASSES = range(self._NUM_CLASSES)

        self._DEVICE = None
        self._LOCAL_RANK = None
        self._LOAD_SUCCESS: bool = False
        try:
            if self.DDP_ON:
                self._LOCAL_RANK = int(os.environ["LOCAL_RANK"])
            else:
                self._LOCAL_RANK = 0
        except KeyError:
            self._LOCAL_RANK = 0
            self.DDP_ON = False
            print("Failed to use DDP!")
            
            
        self._DEVICE = torch.device("cuda", self._LOCAL_RANK)
        
        if len(dict_config) != 0:
            d = eval(dict_config[0])
            for k in dict(d):
                setattr(self, k, d[k])

    def save(self, fn='/config.json'):
        with open(self._WORKING_DIR + fn, 'w') as fp:
            json.dump(str(self.__dict__), fp, indent=4)

    def load(self, fn='/config.json'):
        try:

            with open(self._WORKING_DIR + fn, 'r') as fp:
                dict_config = json.load(fp)
                d = eval(dict_config)
                for k in dict(d):
                    setattr(self, k, d[k])
            print("Config file loaded successfully!")
        except:
            print("Config file does not exits, use default value instead!")


configs = Config()
