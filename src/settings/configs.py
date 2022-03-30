import copy
import json
import os

import torch


class Config:
    def __init__(self, *dict_config) -> None:
        # ==============================================
        # GLOBAL SETTINGS

        # Directory right under the root of the project
        self.MODEL_DIR_NAME: str = "/models_v100_new/"
        self.TRAINING_DATA_DIR: str = "/data/"
        self.SUBMISSION_DATA_DIR: str = "/test_images/"

        # ==============================================
        # MAIN TRAINING SETTINGS 
        # WHICH COULD EFFECT MODEL PERFORMANCE

        # Select in optim/load_opt and loss/load_loss
        self.MODEL: str = "resnet50"
        self.OPT: str = "Adam"
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
        self.TEST_N_DATA_POINTS: int = 60000

        # ==============================================
        # MODEL LOADING SETTINGS

        # If load specific failed, 
        # will fall back to use [empty model prototype | load best model]
        # LOAD_BEST_MODEL is prior to LOAD_SPECIFIC_MODEL
        self.LOAD_SPECIFIC_MODEL: bool = True
        self.LOAD_BEST_MODEL: bool = True
        self.MODEL_NAME: str = "53_05.pth"

        # If true, a submission file will be generated before training
        self.GEN_SUBMISSION: bool = False
        self.SUBMISSION_FN: str = 'submit50_new.csv'

        # ==============================================
        # PRIVATE VALUES

        self._NUM_CLASSES: int = 15505
        self._CLASSES = range(self._NUM_CLASSES)

        cur_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self._WORKING_DIR: str = os.path.join('/', *cur_dir.split("/")[:-2])
        self._MODEL_DIR: str = self._WORKING_DIR + self.MODEL_DIR_NAME
        self._DATA_DIR: str = self._WORKING_DIR + self.TRAINING_DATA_DIR
        self._SUBMISSION_DATA_DIR: str = self._WORKING_DIR + self.SUBMISSION_DATA_DIR

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

        if self._LOCAL_RANK == 0 and not os.path.exists(self._MODEL_DIR):
            os.makedirs(self._MODEL_DIR)
        self._DEVICE = torch.device("cuda", self._LOCAL_RANK)

        if len(dict_config) != 0:
            d = eval(dict_config[0])
            for k in dict(d):
                setattr(self, k, d[k])

    def save(self, fn='config.json'):
        if self._LOCAL_RANK != 0:
            return
        with open(self._WORKING_DIR + "/" + fn, 'w') as fp:
            dict_copy = copy.deepcopy(self.__dict__)

            # Remove private properties witch should be derived on-the-fly
            del_key = []
            for i in dict_copy:
                if i.startswith('_'):
                    del_key.append(i)
            for i in del_key:
                del dict_copy[i]

            json.dump(dict_copy, fp, indent=4)
        print(f"Config file successfully saved to {fn}!")

    def load(self, fn='config.json'):

        try:
            with open(self._WORKING_DIR + "/" + fn, 'r') as fp:
                dict_config = json.load(fp)
                for k in dict(dict_config):
                    try:
                        if type(dict_config[k]) == type(getattr(self, k)):
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
                            print(f"Key [\"{k}\"] will be discarded since config class does not use this attribute.")
            if self._LOCAL_RANK == 0:
                print(f"Config file {fn} loaded successfully!")

        except:
            if self._LOCAL_RANK == 0:
                print(f"Config file {fn} failed to load! Use default value instead!")


configs = Config()
