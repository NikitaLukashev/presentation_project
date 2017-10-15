#! /usr/bin/env python3
# coding: utf-8

from sklearn.model_selection import StratifiedKFold


# global variable
path = '/home/nikita/tools/PycharmProjects/presentation_project'
n_folds = 5
my_seed = 0
my_cv= StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=my_seed)
