#!/usr/bin/python3.5
import sys
import pandas as pd
import numpy as np
import settings
from sklearn.linear_model import LinearRegression

# feature params
ALL_FEATURES = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday',
                'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance',
                'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                'Promo2SinceYear', 'PromoInterval', 'DateOfCompetitionOpen', 'IsInCompetition']
TRAIN_FEATURES = []
TEST_FEATURES = list(TRAIN_FEATURES)


def remove_closed_stores(df):
    """
    Returns a pd.DataFrame without closed stores.
    (closed means 'Open' == 0)
    :param df:
    :return: a DataFrame without closed stores.
    """
    return df[df['Open'] != 0]


def one_hot_encode(df, target, prefix, excludes=None):
    """
    Perform one-hot encoding for a target column in a given pd.DataFrame
    :param df:
    :param target: (str) name of the column to one-hot encode
    :param prefix: (str) prefix of the encoding variables
    :param excludes: (list) of encoded column names to drop (including prefix)
    :return:
    """
    dummy = pd.get_dummies(df[target], prefix=prefix)
    if excludes is not None:
        dummy.drop(excludes, axis=1, inplace=True)
    df.drop([target], axis=1, inplace=True)
    return df.join(dummy)


def generate_month_feature(df):
    df['Month'] = df['Date'].str[5:7]
    df['Month'] = df['Month'].astype(int)
    return df


def prepare_data(df, to_drop, has_y=False):
    """
    Performs the common train/test feature engineering procedures
    :param df:
    :param to_drop: (list) of column names to drop from both train and test sets
    :param has_y: (bool) does the df contain the y variable of interest?
    :return:
    """
    # df = generate_month_feature(df)

    # drop features we don't need
    df = df.drop(to_drop, axis=1)

    # recode StateHoliday to something numeric
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df['StateHoliday'] = df['StateHoliday'].replace(['0', 'a', 'c'], [0, 1, 2])

    # norm customers using log_2
    df['Customers'] = df['Customers'].replace([0], [1])  # prevent nans
    df['Customers'] = np.log2(df['Customers'])

    # one-hot encode dayofweek
    df = one_hot_encode(df, 'DayOfWeek', 'Day', ['Day_7'])

    # append Id column
    df['Id'] = df.index

    # normalize sales using log_2 (only applicable for training data)
    if has_y:
        df['Sales'] = df['Sales'].replace([0], [1])  # prevent nans
        df['Sales'] = np.log2(df['Sales'])

    return df


def extract_closed_store_ids(df):
    """
    Gets a list of IDs for stores that are closed in df
    :param df:
    :return: the IDs of stores that are closed in df
    """
    return df[df['Open'] == 0]['Id']


train = pd.read_csv(settings.CSV_TRAIN, low_memory=False)
test = pd.read_csv(settings.CSV_TEST, low_memory=False)
store = pd.read_csv(settings.CSV_STORE, low_memory=False)

train = prepare_data(train, to_drop=['Date'], has_y=True)
train = remove_closed_stores(train)

test = prepare_data(test, to_drop=['Date'], has_y=False)


def split_open_closed(test_df):
    closed_store_ids = extract_closed_store_ids(test_df)
    test_df = remove_closed_stores(test_df)
    test_df = test_df.drop(['Open'], axis=1)
    return closed_store_ids, test_df


def save_for_submission_csv(closed_store_ids, open_store_sales, outfile):
    open_store_sales = pd.DataFrame(
        {'Id': open_store_sales.index + 1, 'Sales': np.power(2, open_store_sales.values)})
    closed_store_sales = pd.DataFrame(
        {'Id': closed_store_ids + 1, 'Sales': 1})  # 0 sales need to be map to 1 for kaggle
    submission = pd.concat([open_store_sales, closed_store_sales])
    submission.to_csv(outfile, index=False)
    return


def train_many_rf_models(train_df, test_df, outfile='output.csv', verbose=False):
    """
    Trains a random forest for each store and generates predictions for all testing input
    :param train_df: processed dataframe of training data
    :param test_df: processed dataframe of testing data (should not have Sales column)
    :param closed_store_ids: list of stores that are closed in the testing data
    :param outfile: name of the output file to write predictions to
    :return:
    """
    print_feature_list = True
    print('\nTraining many RF model')
    # special case for closed stores where sales = 0 (or 1 for kaggle's purposes)
    closed_store_ids, test_df = split_open_closed(test_df)

    train_stores = dict(list(train_df.groupby('Store')))
    test_stores = dict(list(test_df.groupby('Store')))
    open_store_sales = pd.Series()
    train_scores = []
    for i in test_stores:
        current_store = train_stores[i]

        # define training and testing sets
        train_x = current_store.drop(['Id', 'Sales', 'Store', 'Open'], axis=1)

        if print_feature_list:
            print('Features: %s' % train_x.columns.values.tolist())
            print_feature_list = False

        train_y = current_store['Sales']

        test_x = test_stores[i].copy()
        open_store_ids = test_x['Id']
        test_x = test_x.drop(['Id', 'Store'], axis=1)

        model = LinearRegression()
        model.fit(train_x, train_y)
        test_y = model.predict(test_x)
        train_scores.append(model.score(train_x, train_y))

        # append predicted values of current store to submission
        open_store_sales = open_store_sales.append(pd.Series(test_y, index=open_store_ids))

        if verbose:
            print('Completed Store %d: train_score=%.5f' % (i, train_scores[-1]))

    # save to csv file
    save_for_submission_csv(closed_store_ids, open_store_sales, outfile)
    print('mean(train_score)=%.5f' % np.mean(train_scores))
    print('sd(train_score)=%.5f' % np.std(train_scores))
    print('done: wrote predictions to %s' % outfile)
    return


def train_monolithic_rf_model(train_df, test_df, outfile='out.csv'):
    print('\nTraining monolithic RF model')
    # special case for closed stores where sales = 0 (or 1 for kaggle's purposes)
    closed_store_ids, test_df = split_open_closed(test_df)

    train_y = train_df['Sales']
    train_x = train_df.drop(['Id', 'Sales', 'Open'], axis=1)

    open_store_ids = test_df['Id']
    test_x = test_df.drop(['Id'], axis=1)

    print('Features: %s' % train_x.columns.values.tolist())
    model = LinearRegression()
    model.fit(train_x, train_y)

    test_y = model.predict(test_x)
    train_score = model.score(train_x, train_y)

    open_store_sales = pd.Series().append(pd.Series(test_y, index=open_store_ids))

    # save to csv file
    save_for_submission_csv(closed_store_ids, open_store_sales, outfile)
    print('train_score=%.5f' % train_score)
    print('done: wrote predictions to %s' % outfile)
    return


train_many_rf_models(train, test, outfile='rossmann-rf-per-store.csv', verbose=False)
train_monolithic_rf_model(train, test, outfile='rossmann-monolithic-rf.csv')
