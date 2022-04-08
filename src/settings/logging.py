import logging
from logging.config import dictConfig
import os

class SupperessReducerInfo(logging.Filter):
    def filter(self, record):
        
        try:
            if int(os.environ["LOCAL_RANK"]) != 0:
                return False
        except:
            return False
        
        if record.filename == "distributed.py" and record.funcName == "forward":
            return False
        return True

LOGGING_CONFIG = { 
    'version': 1,
    'disable_existing_loggers': True,
    'filters': {
        'filter': {
            '()': SupperessReducerInfo,
        }
    },
    'formatters': {
        'console_fmt': { 
            'format': '%(asctime)s %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'file_fmt': { 
            'format': '%(asctime)s %(levelname)-8s %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'stream_handler': { 
            'level': 'INFO',
            'formatter': 'console_fmt',
            'class': 'logging.StreamHandler',
        },
        'file_handler': { 
            'level': 'INFO',
            'formatter': 'file_fmt',
            'class': 'logging.FileHandler',
            'filename': 'test1.log',
            'mode': 'a'
        },
    },
    'loggers': { 
        '': {  # root logger
            'handlers': ['stream_handler', 'file_handler'],
            'level': 'INFO',
            'propagate': False,
            "filters": ["filter"]
        }
    } 
}

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger()
