import pandas as pd
import numpy as np
import lightgbm as lgbm
import scipy
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import optuna
from typing import List, Dict, Tuple, Union
import joblib
import os


def train(X_train: pd.DataFrame, y_train: pd.Series, params: Dict={}, random_state: int=None):
        '''
        Train model
        Args:
            X_train: training features
            y_train: response variable
            params: a dictionary of lgbm params
        Returns:
            trained model
        '''
        model = lgbm.LGBMClassifier(**params, seed=random_state)
        model.fit(X_train, y_train)
        return model

def eval_params(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict={},
    random_state: int=None
    ):
    '''
    train model with train set, and evaluate with validation set
    Args:
        X_train: training features
        y_train: training target
        X_val: validation features
        y_val: validation target
        params: a dictionary of lgbm params
        random_state: seed
    Returns:
        ROC AUC score
    '''
    model = train(X_train, y_train, params, random_state=random_state)
    pred = model.predict_proba(X_val)
    return roc_auc_score(y_val, pred[:, 1])


def eval_params_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict={},
    random_state: int=None
    ):
    '''
    5-fold stratified CV (see function "eval_params")
    Returns:
        average score
    '''
    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_val = X.iloc[val_index]
        y_val = y.iloc[val_index]
        score = eval_params(X_train, y_train, X_val, y_val, params, random_state=random_state)
        scores.append(score)
    return np.mean(scores)


def preprocess(
    df: pd.DataFrame,
    continuous_columns: List[str],
    ordinal_columns: List[str],
    date_columns: List[str],
    target_column: str
    ):
    '''
    preprocess dataset
    Returns:
        X and y
    '''
    df_ = pd.DataFrame()

    # do nothing with continuous variables
    df_[continuous_columns] = df[continuous_columns]
    # encode ordinal variables
    encoder = OrdinalEncoder()
    encoder.fit(df[ordinal_columns])
    df_[ordinal_columns] = encoder.transform(df[ordinal_columns])
    #feature engineering for date columns - hour, day of week, day, month
    for c in date_columns:
        df_[f"{c}_hod"] = df[c].dt.hour
        df_[f"{c}_dow"] = df[c].dt.weekday
        df_[f"{c}_day"] = df[c].dt.day
        df_[f"{c}_month"] = df[c].dt.month
        df_[f"{c}_year"] = df[c].dt.year

    return df_, df[target_column]


def get_lgbm_optimized_params(
    df: pd.DataFrame,
    continuous_columns: List[str],
    ordinal_columns: List[str],
    date_columns: List[str],
    target_column: str,
    save_dir: str,
    hyperparams_stepwise_groups: List[List[str]]=None,
    fixed_params: Dict={},
    n_trials: int=100,
    random_state: int=None
    ):
    '''
    Perform bayesian optimization of parameters
    Args:
        df: Pandas DataFrame will columns of features including target/label
        continuous_columns: list of column names of continuous features
        ordinal_columns: list of column names of features to be ordinally encoded
        date_columns: list of column names of dates from which temporal features are to be generated
            generated features: hour, day of week, day, month
        target_column: column name of target/response variable/dependant variables
        hyperparams_stepwise_groups: grouping of hyperparams to be jointly and stepwisely optimized
            format:
                [
                    [<param_1.1>, <param_1.2>, ...],
                    [<param_2.1>, <param_2.2>, ...],
                    ...
                ]
            all <param_1.j> will be optimized jointly, followed by all <param_2.j>, etc.
        fixed_params: fixed lgbm params, specified but not optimized
        n_trials: n optimziation iterations
        random_state: seed
    Returns:
        A tuple of best params and best score
    '''
    # create save dir
    os.makedirs(save_dir, exist_ok=True)

    X, y = preprocess(df, continuous_columns, ordinal_columns, date_columns, target_column)

    # TRAINING LOOP
    spec = {
        'n_estimators': {'type': 'int', 'lb': 50, 'ub': 1000, 'log': False, 'default': 100},
        'learning_rate': {'type': 'float', 'lb': 0.001, 'ub': 1.0, 'log': True, 'default': 0.1},
        'lambda_l1': {'type': 'float', 'lb': 1e-8, 'ub': 10.0, 'log': True, 'default': 1e-8},
        'lambda_l2': {'type': 'float', 'lb': 1e-8, 'ub': 10.0, 'log': True, 'default': 1e-8},
        'num_leaves': {'type': 'int', 'lb': 2**3, 'ub': 2**8, 'log': False, 'default': 31},
        'feature_fraction': {'type': 'float', 'lb': 0.4, 'ub': 1.0, 'log': False, 'default': 1.0},
        'bagging_fraction': {'type': 'float', 'lb': 0.4, 'ub': 1.0, 'log': False, 'default': 1.0},
        'bagging_freq': {'type': 'int', 'lb': 1, 'ub': 7, 'log': False, 'default': 0},
        'min_child_samples': {'type': 'int', 'lb': 5, 'ub': 100, 'log': False, 'default': 20},
    }

    if hyperparams_stepwise_groups is None:
        hyperparams_stepwise_groups = [[param for param in spec]]
    
    best_params = {}    # to collect optimized params
    best_params.update(fixed_params)
    for i, group in enumerate(hyperparams_stepwise_groups):

        print(f"optimizing group: {group}")
        
        def objective(trial):
            
            params = {}
            for param in group:
                if spec[param]['type'] == 'int':
                    params[param] = trial.suggest_int(param, spec[param]['lb'], spec[param]['ub'], log=spec[param]['log'])
                else:
                    params[param] = trial.suggest_float(param, spec[param]['lb'], spec[param]['ub'], log=spec[param]['log'])

            # update with best_params so far
            params.update(best_params)

            return eval_params_cv(X, y, params=params, random_state=random_state)

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.enqueue_trial({param: spec[param]['default'] for param in group})    # insert default value for the first optimization iteration
        study.optimize(objective, n_trials=n_trials)
        best_params.update(study.best_params)
        print(f"best params: {best_params}, best score: {study.best_value}")

        # save study
        joblib.dump(study, f"{save_dir}/study_{i}_{group}_best_score_{study.best_value}.pkl")
    
    print('optimization complete')
    return best_params, study.best_value
