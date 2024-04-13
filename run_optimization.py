import argparse
from util import *
import lightgbm as lgbm
import joblib
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/data.csv', help="dataset path")
    parser.add_argument('--save_dir', type=str, default='runs/example_run', help="Directory to save optimization runs and best model")
    parser.add_argument('--config_path', type=str, default='runs/example_run/example_config.json', help="training and optimization config filepath")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    

