#! /usr/bin/env python3
# coding: utf-8

import lightgbm as lgb
import xgboost as xgb
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import cross_val_score


from settings import *


# xgb cv
def xgb_cv(train_df, target_df, xgb_params):
    dtrain = xgb.DMatrix(csr_matrix(train_df), label=target_df)

    res = xgb.cv(xgb_params, dtrain, num_boost_round=100000, folds=my_cv,
                 early_stopping_rounds=200, maximize=False,
                 verbose_eval=200, show_stdv=True)

    best_round = res.shape[0] - 1

    cv_mean = np.round(res.iloc[-1, 0], 4)
    cv_std = np.round(res.iloc[-1, 1], 4)

    print('\n')
    try:
        cv_100_round_mean = np.round(res.iloc[100, 0], 4)
        cv_100_round_std = np.round(res.iloc[100, 1], 4)
        print('#', '[100] {0}+{1}'.format(cv_100_round_mean, cv_100_round_std))
    except IndexError:
        pass

    try:
        cv_200_round_mean = np.round(res.iloc[200, 0], 4)
        cv_200_round_std = np.round(res.iloc[200, 1], 4)
        print('#', '[200] {0}+{1}'.format(cv_200_round_mean, cv_200_round_std))
    except IndexError:
        pass

    print('#', n_folds, 'fold-CV of xgboost : {0}+{1} @ seed={2}, eta={3} , bestRound={4}'
          .format(cv_mean, cv_std, my_seed, xgb_params["learning_rate"], best_round))
    print('\n', '\n')

    return best_round


# lgb cv
def lgb_cv(train_df, target_df, lgb_params):
    dtrain = lgb.Dataset(csr_matrix(train_df), label=target_df)

    res = lgb.cv(lgb_params, dtrain, num_boost_round=10000,
                 folds=my_cv.split(train_df, target_df), early_stopping_rounds=400,
                 verbose_eval=200, show_stdv=True)

    best_round = len(res['multi_logloss-mean']) - 1

    cv_mean = np.round(res['multi_logloss-mean'][-1], 4)
    cv_std = np.round(res['multi_logloss-stdv'][-1], 4)

    print('\n')

    try:
        cv_100_round_mean = np.round(res['multi_logloss-mean'][100 - 1], 4)
        cv_100_round_std = np.round(res['multi_logloss-stdv'][100 - 1], 4)
        print('#', '[100] {0}+{1}'.format(cv_100_round_mean, cv_100_round_std))
    except IndexError:
        pass

    try:
        cv_200_round_mean = np.round(res['multi_logloss-mean'][200 - 1], 4)
        cv_200_round_std = np.round(res['multi_logloss-stdv'][200 - 1], 4)
        print('#', '[200] {0}+{1}'.format(cv_200_round_mean, cv_200_round_std))
    except IndexError:
        pass

    print('#', n_folds, 'fold-CV of lightgbm : {0}+{1} @ seed={2}, eta={3} , bestRound={4}'
          .format(cv_mean, cv_std, my_seed, lgb_params["learning_rate"], best_round))
    print('\n', '\n')

    return best_round


#sk_cv
def sk_cv(train_df, target_df, clf):
    train_df = train_df.fillna(-100)
    tmp=-cross_val_score(clf, train_df, target_df, scoring='neg_log_loss', cv=my_cv, n_jobs=4)
    cv_mean = np.mean(tmp)
    cv_std = np.std(tmp)
    print('#',  n_folds, 'fold-CV {0}+{1}'.format(np.round(cv_mean, 4), np.round(cv_std, 4)))
    return


# xgb predict
def get_xgb_prediction(number_round, params, train_df, test_df, target_df):
    d_train = xgb.DMatrix(csr_matrix(train_df), label=target_df)
    d_test = xgb.DMatrix(csr_matrix(test_df))
    clf = xgb.train(params, d_train, number_round)
    preds = clf.predict(d_test)
    return preds


# lgb predict
def get_lgb_prediction(number_round, params, train_df, test_df, target_df):
    d_train = lgb.Dataset(csr_matrix(train_df), label=target_df)
    clf = lgb.train(params, d_train, number_round)
    preds = clf.predict(test_df)
    return preds
