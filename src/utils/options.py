import argparse
import sys


def get_options(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Parse command for herbarium.")
    parser.add_argument("-f", "--config", type=str, default=None, help="Configuration file location.")
    parser.add_argument("-s", "--savecopy", type=bool, default=False,
                        help="Save a copy of current default configuration.")
    options = parser.parse_args(args)
    return options
