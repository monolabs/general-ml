import argparse
from util import *
import lightgbm as lgbm
import joblib
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='data/dummy_train.csv', help="train dataset path")
    parser.add_argument('--test_data_path', type=str, default='data/dummy_test.csv', help="test dataset path")
    parser.add_argument('--save_dir', type=str, default='runs/example_run', help="Directory to save optimization runs and best model")
    parser.add_argument('--config_path', type=str, default='runs/example_run/example_config.json', help="training and optimization config filepath")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    df_train = pd.read_csv(args.train_data_path)
    df_test = pd.read_csv(args.test_data_path)
    
    # hyperparams optimization
    best_params, best_value = get_lgbm_optimized_params(
        df_train,
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
    
    # training and saving of best model
    print("training best model...")
    X_train, y_train = preprocess(
        df_train, 
        config['continuous_columns'], 
        config['ordinal_columns'], 
        config['date_columns'], 
        config['target_column']
        )
    model = train(X_train, y_train, params=best_params, random_state=config['random_state'])
    joblib.dump(model, f'{args.save_dir}/best_model.pkl')

    # testing
    print("testing...")
    X_test, y_test = preprocess(
        df_train, 
        config['continuous_columns'], 
        config['ordinal_columns'], 
        config['date_columns'], 
        config['target_column']
        )
    y_pred = model.predict_proba(X_test)[:, -1]
    auc_score = roc_auc_score(y_test, y_pred)

    # recording of results
    results = {
        "train_data_path": args.train_data_path,
        "test_data_path": args.test_data_path,
        "config_path": args.config_path,
        "best_value_train": best_value,
        "test_score": auc_score
    }
    with open(f'{args.save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("run complete.")


