#! /usr/bin/env python3
# coding: utf-8


import re
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import os
import gc

from xgb_lgb_utils import *
from features_utils import *


# my function
def read_df(path):
    train = pd.read_json(path + '/train.json', convert_dates=['created'])
    test = pd.read_json(path + '/test.json', convert_dates=['created'])
    submission = pd.read_csv(path + '/sample_submission.csv')
    leaked_feature = pd.read_csv(path + '/listing_image_time.csv')
    return train, test, submission, leaked_feature


def numerise_target(train):
    train['interest_level'] = train['interest_level'].map({'low': 0, 'medium': 1, 'high': 2})
    return train


def get_submission(predictions, model):
    to_submit = pd.DataFrame(data={"listing_id": test.listing_id.values,
                                   "high": predictions[:, 2:3].ravel(),
                                   "medium": predictions[:, 1:2].ravel(),
                                   "low": predictions[:, 0:1].ravel()})
    to_submit.to_csv(os.path.join(path, 'single_model_submission')+"/submission_" + model + '.csv', index=False)


def get_manager_skill(directory):
    random.seed(2017)
    train_d = pd.read_json(os.path.join(directory, 'db') + '/train.json')
    test_d = pd.read_json(os.path.join(directory, 'db') + '/test.json')
    features_to_use = ["listing_id"]

    index = list(range(train_d.shape[0]))
    random.shuffle(index)
    a = [np.nan] * len(train_d)
    b = [np.nan] * len(train_d)
    c = [np.nan] * len(train_d)

    for i in range(5):
        building_level = {}
        for j in train_d['manager_id'].values:
            building_level[j] = [0, 0, 0]
        test_index = index[int((i * train_d.shape[0]) / 5):int(((i + 1) * train_d.shape[0]) / 5)]
        train_index = list(set(index).difference(test_index))
        for j in train_index:
            temp = train_d.iloc[j]
            if temp['interest_level'] == 'low':
                building_level[temp['manager_id']][0] += 1
            if temp['interest_level'] == 'medium':
                building_level[temp['manager_id']][1] += 1
            if temp['interest_level'] == 'high':
                building_level[temp['manager_id']][2] += 1
        for j in test_index:
            temp = train_d.iloc[j]
            if sum(building_level[temp['manager_id']]) != 0:
                a[j] = building_level[temp['manager_id']][0] * 1.0 / sum(building_level[temp['manager_id']])
                b[j] = building_level[temp['manager_id']][1] * 1.0 / sum(building_level[temp['manager_id']])
                c[j] = building_level[temp['manager_id']][2] * 1.0 / sum(building_level[temp['manager_id']])
    train_d['manager_level_low'] = a
    train_d['manager_level_medium'] = b
    train_d['manager_level_high'] = c

    a = []
    b = []
    c = []
    building_level = {}
    for j in train_d['manager_id'].values:
        building_level[j] = [0, 0, 0]
    for j in range(train_d.shape[0]):
        temp = train_d.iloc[j]
        if temp['interest_level'] == 'low':
            building_level[temp['manager_id']][0] += 1
        if temp['interest_level'] == 'medium':
            building_level[temp['manager_id']][1] += 1
        if temp['interest_level'] == 'high':
            building_level[temp['manager_id']][2] += 1

    for i in test_d['manager_id'].values:
        if i not in building_level.keys():
            a.append(np.nan)
            b.append(np.nan)
            c.append(np.nan)
        else:
            a.append(building_level[i][0] * 1.0 / sum(building_level[i]))
            b.append(building_level[i][1] * 1.0 / sum(building_level[i]))
            c.append(building_level[i][2] * 1.0 / sum(building_level[i]))
    test_d['manager_level_low'] = a
    test_d['manager_level_medium'] = b
    test_d['manager_level_high'] = c

    features_to_use.append('manager_level_low')
    features_to_use.append('manager_level_medium')
    features_to_use.append('manager_level_high')

    train_d = train_d[features_to_use]
    test_d = test_d[features_to_use]

    res_train_test = pd.concat((train_d, test_d), axis=0)

    return res_train_test


def try_and_find_nr(description):
    if reg.match(description) is None:
        return 0
    return 1


def featurefixer(l):
    l = [s.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('*', ' ') for s in l]
    s = " ".join(l)
    return s

def time_long(x, y):
    if x == 4:
        return y
    if x == 5:
        return 30 + y
    if x == 6:
        return 30 + 31 + y




if __name__ == '__main__':

    ###################################################################
    #############################load data#############################
    ###################################################################

    # load train, test, submission, leaked_features
    train, test, submission, leaked_feature = read_df(os.path.join(path, 'db'))
    ntrain = train.shape[0]
    train_id = train.index.values
    test_id = test.index.values

    train = numerise_target(train)
    target = train.interest_level
    train.drop(['interest_level'], axis=1, inplace=True)

    # concat train,  test
    train_test = pd.concat((train, test), axis=0)
    del train, test
    gc.collect()

    ###################################################################
    ##########################create features##########################
    ###################################################################

    # some features
    train_test['num_nr_of_lines'] = train_test['description'].apply(lambda x: x.count('<br /><br />'))
    train_test['num_redacted'] = train_test['description'].apply(lambda x: 1 if 'website_redacted' in x else 0)
    train_test['num_email'] = train_test['description'].apply(lambda x: 1 if '@' in x else 0)

    reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
    train_test['num_phone_nr'] = train_test['description'].apply(try_and_find_nr)

    train_test['features'] = train_test["features"].apply(lambda x: featurefixer(x))
    train_test['bathrooms'] = train_test['bathrooms'].replace([20], 2)
    train_test['is_half_bathrooms'] = train_test['bathrooms'] % 1
    train_test['is_5*half_bathrooms'] = train_test['bathrooms'] % 3
    train_test['is_7*half_bathrooms'] = train_test['bathrooms'] % 5

    # date features
    train_test['created'] = pd.to_datetime(train_test['created'])
    train_test['Month'] = train_test['created'].dt.month
    train_test['Week'] = train_test['created'].dt.week
    train_test['Day'] = train_test['created'].dt.day
    train_test['hour'] = train_test['created'].dt.hour
    train_test['minute'] = train_test['created'].dt.minute
    train_test['second'] = train_test['created'].dt.second
    train_test['day_of_year'] = train_test['created'].dt.dayofyear
    train_test['day_of_week'] = train_test['created'].dt.dayofweek

    # other features
    train_test['has_description'] = train_test['description'].apply(
        lambda x: 0 if len(x.strip()) == 0 else 1)
    train_test = get_count(train_test, 'created', ['building_id'])
    train_test = get_count(train_test, 'created', ['manager_id'])
    train_test = get_count(train_test, 'listing_id', ['created'])

    # agregate by building_id
    train_test = get_mean(train_test, 'price', ['building_id'])
    train_test = get_median(train_test, 'price', ['building_id'])
    train_test['price_divided_median_of_price_by_[\'building_id\']'] = train_test['price'] / train_test[
        'median_of_price_by_[\'building_id\']']
    train_test = get_std(train_test, 'price', ['building_id'])

    # other features
    train_test = get_median(train_test, 'price', ['bedrooms'])
    train_test['price_divided_median_of_price_by_bed'] = train_test['price'] / train_test[
        'median_of_price_by_[\'bedrooms\']']
    train_test = get_median(train_test, 'price', ['bathrooms'])

    # agregate by date
    train_test = get_mean(train_test, 'price', ['Month'])

    # agregate by manager_id
    train_test = get_median(train_test, 'price', ['manager_id'])
    train_test = get_sum(train_test, 'price', ['manager_id'])
    train_test = get_std(train_test, 'price', ['manager_id'])

    train_test = get_mean(train_test, 'bathrooms', ['manager_id'])
    train_test = get_median(train_test, 'bathrooms', ['manager_id'])
    train_test = get_sum(train_test, 'bathrooms', ['manager_id'])
    train_test = get_std(train_test, 'bathrooms', ['manager_id'])

    train_test = get_mean(train_test, 'bedrooms', ['manager_id'])
    train_test = get_median(train_test, 'bedrooms', ['manager_id'])
    train_test = get_std(train_test, 'bedrooms', ['manager_id'])

    train_test = get_mean(train_test, 'latitude', ['manager_id'])
    train_test = get_median(train_test, 'latitude', ['manager_id'])
    train_test = get_std(train_test, 'latitude', ['manager_id'])

    train_test = get_mean(train_test, 'longitude', ['manager_id'])
    train_test = get_median(train_test, 'longitude', ['manager_id'])
    train_test = get_sum(train_test, 'longitude', ['manager_id'])
    train_test = get_std(train_test, 'longitude', ['manager_id'])

    train_test = get_mean(train_test, 'Day', ['manager_id'])
    train_test = get_median(train_test, 'Day', ['manager_id'])
    train_test = get_sum(train_test, 'Day', ['manager_id'])
    train_test = get_std(train_test, 'Day', ['manager_id'])

    train_test = get_mean(train_test, 'hour', ['manager_id'])
    train_test = get_median(train_test, 'hour', ['manager_id'])
    train_test = get_sum(train_test, 'hour', ['manager_id'])
    train_test = get_std(train_test, 'hour', ['manager_id'])

    train_test = get_mean(train_test, 'minute', ['manager_id'])
    train_test = get_median(train_test, 'minute', ['manager_id'])
    train_test = get_sum(train_test, 'minute', ['manager_id'])
    train_test = get_std(train_test, 'minute', ['manager_id'])

    train_test = get_mean(train_test, 'second', ['manager_id'])
    train_test = get_median(train_test, 'second', ['manager_id'])
    train_test = get_sum(train_test, 'second', ['manager_id'])
    train_test = get_std(train_test, 'second', ['manager_id'])

    train_test = get_mean(train_test, 'day_of_year', ['manager_id'])
    train_test = get_median(train_test, 'day_of_year', ['manager_id'])
    train_test = get_min(train_test, 'day_of_year', ['manager_id'])
    train_test = get_sum(train_test, 'day_of_year', ['manager_id'])
    train_test = get_std(train_test, 'day_of_year', ['manager_id'])

    train_test = get_mean(train_test, 'day_of_week', ['manager_id'])
    train_test = get_median(train_test, 'day_of_week', ['manager_id'])
    train_test = get_sum(train_test, 'day_of_week', ['manager_id'])

    # photo features
    train_test["num_photos"] = train_test["photos"].apply(len)

    # other features
    train_test["num_features"] = train_test["features"].apply(len)
    train_test["price_bed"] = train_test["price"] / (train_test["bedrooms"] + 1)
    train_test["price_bath"] = train_test["price"] / (train_test["bathrooms"] + 1)
    train_test["bed_bath_per"] = train_test["bedrooms"] / train_test["bathrooms"]
    train_test = get_distinct_count(train_test, 'building_id', ['manager_id'])

    # the range place manager active
    manager_place = {}
    manager_id = train_test["manager_id"].value_counts()

    for man in list(manager_id.index):
        la = train_test[train_test["manager_id"] == man]["latitude"].copy()
        lo = train_test[train_test["manager_id"] == man]["longitude"].copy()
        manager_place[man] = 10000 * ((la.max() - la.min()) * (lo.max() - lo.min()))

    train_test["manager_place"] = list(map(lambda x: manager_place[x], train_test["manager_id"]))  # worse
    train_test["midu"] = train_test['distinct_count_of_building_id_by_[\'manager_id\']'] / train_test[ "manager_place"]
    train_test.drop(['manager_place'], axis=1, inplace=True)


    # forum features
    train_test["time"] = list(map(lambda x, y: time_long(x, y), train_test["Month"], train_test["Day"]))
    train_test["all_hours"] = train_test["time"] * 24 + train_test["hour"]

    train_test = get_distinct_count(train_test, 'time', ['manager_id'])
    train_test = get_distinct_count(train_test, 'all_hours', ['manager_id'])
    train_test["manager_hours_rt"] = train_test["distinct_count_of_all_hours_by_[\'manager_id\']"] / train_test["distinct_count_of_time_by_[\'manager_id\']"]

    # tfidf features
    train_test = get_tfidf_vectorizer(train_test, 'display_address',
                                      TfidfVectorizer(analyzer='word', max_features=25, ngram_range=(1, 1),
                                                      token_pattern='^[a-z]+$'))
    train_test = get_tfidf_vectorizer(train_test, 'description',
                                      TfidfVectorizer(analyzer='word', max_features=10, stop_words='english',
                                                      use_idf=True, binary=False, ngram_range=(3, 3),
                                                      token_pattern=r'(?u)\b\w\w+\b', norm='l2'))
    train_test = get_tfidf_vectorizer(train_test, 'features',
                                      TfidfVectorizer(analyzer='word', max_features=50, stop_words='english',
                                                      use_idf=False, binary=False, ngram_range=(1, 1),
                                                      token_pattern=r'(?u)\b\w\w+\b', norm=None))

    # label encoded features
    le = preprocessing.LabelEncoder()
    train_test['building_id'] = le.fit_transform(train_test['building_id'])  # worse
    train_test['manager_id'] = le.fit_transform(train_test['manager_id'])
    train_test['display_address'] = le.fit_transform(train_test['display_address'])
    train_test['street_address'] = le.fit_transform(train_test['street_address'])
    train_test['mean_of_price_by_building_id_minus_price'] = train_test['mean_of_price_by_[\'building_id\']'] - train_test['price']

    # interaction features
    train_test['bathroomsMultiplyLatitude'] = train_test['bathrooms'] * train_test['latitude']
    train_test['bathroomsMultiplyListing_id'] = train_test['bathrooms'] * train_test['listing_id']
    train_test['bedroomsMinusLongitude'] = train_test['bedrooms'] - train_test['longitude']
    train_test['bedroomsDivideLongitude'] = train_test['bedrooms'] / train_test['longitude']
    train_test['latitudeMinuslongitude'] = train_test['latitude'] - train_test['longitude']
    train_test['latitudeFoislongitude'] = train_test['latitude'] * train_test['longitude']
    train_test['latitudeFDividedlongitude'] = train_test['latitude'] / train_test['longitude']
    train_test['priceDivideBathrooms'] = train_test['price'] / (1 + train_test['bathrooms'])

    # merge train_test with leaked
    train_test = train_test.merge(leaked_feature, left_on='listing_id', right_on='Listing_Id', how='left')

    # spacial features
    train_test['distance_to_median'] = np.sqrt(
        np.power(train_test['latitude'] - np.median(train_test['latitude']), 2) + np.power(
            train_test['longitude'] - np.median(train_test['longitude']), 2))

    # lag features
    train_test['groupe_3'] = train_test['manager_id'] * train_test['latitude'] * train_test['longitude']

    train_test.sort_values(by=['groupe_3', 'created'], axis=0, ascending=True, inplace=True)
    train_test['diff_lag_price_3'] = - train_test['price'].shift(1) + train_test['price']
    train_test.drop(['groupe_3'], axis=1, inplace=True)
    train_test.sort_index(inplace=True)

    # remove worse and non numerical features
    train_test.drop(['building_id', 'created', 'description', 'features', 'photos', 'Month', 'time', "all_hours", "distinct_count_of_time_by_[\'manager_id\']"], axis=1, inplace=True)

    # replace useless values
    train_test.replace([-np.inf, np.inf, np.nan], [999, 999, 999], inplace=True)

    # get manager skill features from forum
    res = get_manager_skill(path)
    train_test = pd.merge(train_test, res, how='left', left_on='listing_id', sort=False, right_on='listing_id')

    # get train, test, release memory
    train = train_test.iloc[:ntrain, :]
    test = train_test.iloc[ntrain:, :]

    del train_test
    gc.collect()


    ###################################################################
    ##########################run cv, predict, write submission########
    ###################################################################

    # lgb cv
    lgb_params = {
        'application': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 3,
        'num_threads': 4,
        'verbose': -1,
        'learning_rate': 0.01,  # 0.01
        'feature_fraction': 0.25,  # 0.2
        'bagging_fraction': 0.5,  # don't change
        'num_leaves': 32,  # 32
        'max_bin': 255,  # don t change
        # 'min_data_in_leaf' : 1,
        'max_depth': 8,  # 8
        'boosting_type': 'gbdt',  # gbdt
        'is_sparse':True
    }

    best_lgb_round = lgb_cv(train, target, lgb_params)
    preds = get_lgb_prediction(best_lgb_round, lgb_params, train, test, target)
    get_submission(preds, 'lgb')

    # [100] 0.7522+0.0013
    # [200] 0.6407+0.002
    # 5 fold-CV of lightgbm : 0.5025+0.0025 @ seed=0, eta=0.01 , bestRound=3912


    # xgb cv
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'nthread': 4,
        'colsample_bytree': 0.3,
        'silent': 1,
        'subsample': 1,
        'learning_rate': 0.03,#0.03
        'max_depth': 6,  # 6
        'min_child_weight': 7,
        'lambda': 1,
        'alpha': 5
    }

    best_xgb_round = xgb_cv(train, target, xgb_params)
    preds = get_xgb_prediction(best_xgb_round, xgb_params, train, test, target)
    get_submission(preds, 'xgb')

    # [100] 0.5886+0.0022
    # [200] 0.5432+0.0025
    # 5 fold-CV of xgboost : 0.503+0.0035 @ seed=0, eta=0.03 , bestRound=1275
