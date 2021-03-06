import copy
import json
import os
import time
import shutil
import torch

import argparse
import sys
from .logging import logger, LOGGING_CONFIG
from logging.config import dictConfig

class Config:
    def __init__(self, fn=None) -> None:
        # ==============================================
        # GLOBAL SETTINGS

        # Directory right under the root of the project
        self.TRAINING_DATA_DIR: str = "/data/"
        self.SUBMISSION_DATA_DIR: str = "/test_images/"

        # ==============================================
        # MAIN TRAINING SETTINGS 
        # WHICH COULD EFFECT MODEL PERFORMANCE

        # Select in optim/load_opt and loss/load_loss
        self.MODEL: str = "resnet50"
        self.OPT: str = "Adam"
        self.SKD: str = "MultiStepLR"
        self.LOSS: str = "CrossEntropy"

        self.BATCH_SIZE: int = 512

        self.LEARNING_RATE: float = 1e-2
        self.LEARNING_RATE_DECREASE_EPOCHS: list = [5, 10, 15, 20]
        self.LEARNING_RATE_GAMMA: float = 0.4

        # TRAINING SPEED RELATED SETTINGS AND CUSTOM CONFIGURATION
        # n workers for loading data
        self.NUM_WORKERS: int = 12
        self.DDP_ON: bool = True
        self.MIX_PRECISION: bool = True
        self.TOTAL_EPOCHS: int = 300
        self.EPOCHS_PER_EVAL: int = 1
        # Train with {len(all data) - TEST_N_DATA_POINTS}
        # Test with {TEST_N_DATA_POINTS}
        self.TEST_ON_N_PERCENT_DATA: float = 0.1
        self.TENSOR_BOARD_ON: bool = False
        # ==============================================
        # MODEL LOADING SETTINGS

        # If load specific failed, 
        # will fall back to use [empty model prototype | load best model]
        # LOAD_BEST_MODEL is prior to LOAD_SPECIFIC_MODEL
        self.LOAD_SPECIFIC_MODEL: bool = True
        self.LOAD_BEST_MODEL: bool = True
        self.MODEL_NAME: str = ""

        # If true, a submission file will be generated before training
        self.GEN_SUBMISSION: bool = False
        self.SUBMISSION_FN: str = 'submission.csv'
        
        # Image laoder backend ['PIL', 'accimage']
        self.IMAGE_BACKEND: str = 'accimage'

        # ==============================================
        # PRIVATE VALUES
        # below attributes should all be derived on-the-fly
        
        self._LOCAL_RANK = None
        try:
            if self.DDP_ON:
                self._LOCAL_RANK = int(os.environ["LOCAL_RANK"])
            else:
                self._LOCAL_RANK = 0
        except KeyError:
            self._LOCAL_RANK = 0
            self.DDP_ON = False
            logger.warning("Failed to use DDP!")

        cur_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self._WORKING_DIR: str = os.path.join('/', *cur_dir.split("/")[:-2])
        if fn is not None:
            self.load(fn)
            
        self._CUR_EPOCHS: int = 1
        self._MODEL_DIR_NAME: str = "/models_saved/"
        self._MODEL_DIR: str = self._WORKING_DIR + self._MODEL_DIR_NAME + self.MODEL + '/'

        if self._LOCAL_RANK == 0 and not os.path.exists(self._MODEL_DIR):
            os.makedirs(self._MODEL_DIR)
        time.sleep(0.5)
        
        # Specify the handler filename and apply the config to logger
        LOGGING_CONFIG['handlers']['file_handler']['filename'] = self._MODEL_DIR + self.MODEL + '.log'
        dictConfig(LOGGING_CONFIG)
           
        
        self._DATA_DIR: str = self._WORKING_DIR + self.TRAINING_DATA_DIR
        self._SUBMISSION_DATA_DIR: str = self._WORKING_DIR + self.SUBMISSION_DATA_DIR
        self._EVAL_TMP_DIR = self._WORKING_DIR + "/eval_tmp/"
        if self._LOCAL_RANK == 0 and os.path.exists(self._EVAL_TMP_DIR):
            shutil.rmtree(self._EVAL_TMP_DIR)
        self._DEVICE = None
        self._LOAD_SUCCESS: bool = False
        self._DEVICE = torch.device("cuda", self._LOCAL_RANK)

    def save(self):
        if self._LOCAL_RANK != 0:
            return
        fn = self._MODEL_DIR + f'{configs.MODEL}' + '_config.json'
        with open(fn, 'w') as fp:
            dict_copy = copy.deepcopy(self.__dict__)

            # Remove private properties witch should be derived on-the-fly
            del_key = []
            for i in dict_copy:
                if i.startswith('_'):
                    del_key.append(i)
            for i in del_key:
                del dict_copy[i]

            json.dump(dict_copy, fp, indent=4)
        logger.info(f"Config file successfully saved to {fn}!")

    def load(self, fn='config.json'):

        try:
            with open(self._WORKING_DIR + "/" + fn, 'r') as fp:
                dict_config = json.load(fp)
                for k in dict(dict_config):
                    try:
                        if not isinstance(dict_config[k], type(getattr(self, k))):
                            cur_type = type(getattr(self, k))
                            if self._LOCAL_RANK == 0:
                                print("Warning! Config file contains unmatched value type, "
                                      "could be a broken configuration")
                                print(f"Key [\"{k}\"]: Casting {dict_config[k]} ({type(dict_config[k])}) "
                                      f"to {cur_type(dict_config[k])} ({cur_type})")
                            setattr(self, k, cur_type(dict_config[k]))
                        else:
                            setattr(self, k, dict_config[k])
                    except AttributeError:
                        if self._LOCAL_RANK == 0:
                            logger.warning(f"Key [\"{k}\"] will be discarded since config class does not use this attribute.")
            if self._LOCAL_RANK == 0:
                print(f"Config file {fn} loaded successfully!")

        except Exception as e:
            if self._LOCAL_RANK == 0:
                print(f"Config file {fn} failed to load! Use default value instead!")
                print(e)
                
    def __str__(self):
       return f"{self.MODEL}, {self.OPT}, {self.SKD}, {self.LOSS}, bs={self.BATCH_SIZE}, lr=({self.LEARNING_RATE},{str(self.LEARNING_RATE_DECREASE_EPOCHS)})"

def get_options(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Parse command for herbarium.")
    parser.add_argument("-f", "--config", type=str, default=None, help="Configuration file location.")

    options = parser.parse_known_args(args)
    return options[0]


options = get_options()
configs = Config(options.config)

configs.save()
logger.info(f"Config: {configs}")
