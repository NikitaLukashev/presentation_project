#! /usr/bin/env python3
# coding: utf-8

import pandas as pd


# ohe
def get_dummies_vectorizer(df, field):
    dummies = pd.get_dummies(df[field], prefix=field, dummy_na=True)
    df = pd.concat([df, dummies], axis=1)
    return df


# tfidf
def get_tfidf_vectorizer(df, field, vectorizer):
    tfidf = vectorizer.fit_transform(df[field])
    tfidf_cols = vectorizer.get_feature_names()
    temp = pd.DataFrame(data=tfidf.todense(), columns=['tfidf_' + field + '_' + i for i in tfidf_cols])
    df = pd.concat([df, temp], axis=1)
    print(tfidf_cols)
    return df


# make aggregate of field by by_field on df
def get_mean(df, field, by_field):
    temp = df.groupby(by_field).mean()[[field]].reset_index()
    temp.columns = [by_field, 'mean_of_' + field + '_by_' + by_field]
    df = df.merge(temp, on=by_field, how='left')
    return df


def get_median(df, field, by_field):
    temp = df.groupby(by_field).median()[[field]].reset_index()
    temp.columns = [by_field, 'median_of_' + field + '_by_' + by_field]
    df = df.merge(temp, on=by_field, how='left')
    return df


def get_max(df, field, by_field):
    temp = df.groupby(by_field).max()[[field]].reset_index()
    temp.columns = [by_field, 'max_of_' + field + '_by_' + by_field]
    df = df.merge(temp, on=by_field, how='left')
    return df


def get_min(df, field, by_field):
    temp = df.groupby(by_field).min()[[field]].reset_index()
    temp.columns = [by_field, 'min_of_' + field + '_by_' + by_field]
    df = df.merge(temp, on=by_field, how='left')
    return df


def get_sum(df, field, by_field):
    temp = df.groupby(by_field).sum()[[field]].reset_index()
    temp.columns = [by_field, 'sum_of_' + field + '_by_' + by_field]
    df = df.merge(temp, on=by_field, how='left')
    return df


def get_std(df, field, by_field):
    tmp = df.groupby(by_field).std()[[field]].reset_index()
    tmp.columns = [by_field, 'std_of_' + field + '_by_' + by_field]
    df = df.merge(tmp, on=by_field, how='left')
    return df


def get_count(df, field, by_field):
    tmp=df.copy()
    tmp[field].fillna('xxx', inplace=True)
    tmp = tmp.groupby(by_field).count()[[field]].reset_index()
    tmp.columns = [by_field, 'count_of_' + field + '_by_' + by_field]
    df = df.merge(tmp, on=by_field, how='left')
    return df


def get_distinct_count(df, field, by_field):
    tmp=df.copy()
    tmp[field].fillna('xxx', inplace=True)
    tmp = tmp[[by_field, field]]
    tmp = tmp.drop_duplicates(inplace=False)
    tmp = tmp.groupby(by_field).count()[[field]].reset_index()
    tmp.columns = [by_field, 'distinct_count_of_' + field + '_by_' + by_field]
    df = df.merge(tmp, on=by_field, how='left')
    return df


# make interaction of two field on df
def get_interaction_product(df, field_1, field_2):
    df[field_1 + '_*_' + field_2] = df[field_1] * df[field_2]
    return df


def get_interaction_quotient(df, field_1, field_2):
    df[field_1 + '_/_' + field_2] = df[field_1] / df[field_2]
    return df


def get_interaction_sum(df, field_1, field_2):
    df[field_1 + '_+_' + field_2] = df[field_1] + df[field_2]
    return df


def get_interaction_minus(df, field_1, field_2):
    df[field_1 + '_-_' + field_2] = df[field_1] - df[field_2]
    return df