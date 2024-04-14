import argparse
from util import *
import lightgbm as lgbm
import joblib
import json
import pandas as pd


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
    df = pd.read_csv(args.data_path)
    
    best_params, best_value = get_lgbm_optimized_params(
        df,
        config['continuous_columns'],
        config['ordinal_columns'],
        config['date_columns'],
        config['target_column'],
        args.save_dir,
        hyperparams_stepwise_groups=config['hyperparams_stepwise_groups'],
        fixed_params=config['fixed_params'],
        n_trials=config['n_trials'],
        random_state=config['random_state']
        )

