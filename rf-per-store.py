#!/usr/bin/python3.5
import sys
import pandas as pd
import numpy as np
import settings
from sklearn.linear_model import LinearRegression

# feature params
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


def prepare_data(df, to_drop):
    """
    Performs the common train/test feature engineering procedures
    :param df:
    :param to_drop: (list) of column names to drop from both train and test sets
    :return:
    """
    # drop features we don't need
    df = df.drop(to_drop, axis=1)

    # recode StateHoliday to something numeric
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df['StateHoliday'] = df['StateHoliday'].replace(['0', 'a', 'c'], [0, 1, 2])

    # one-hot encode dayofweek
    df = one_hot_encode(df, 'DayOfWeek', 'Day', ['Day_7'])

    # append Id column
    df['Id'] = df.index

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

train = prepare_data(train, to_drop=['Customers', 'Date'])
test = prepare_data(test, to_drop=['Customers', 'Date'])

# normalize sales
train['Sales'] = train['Sales'].replace([0], [1])  # prevent nans
train['Sales'] = np.log2(train['Sales'])
# print(train.head(n=10))

closed_store_ids = extract_closed_store_ids(test)
test = remove_closed_stores(test)
test = test.drop(['Open'], axis=1)


def train_models(train_df, test_df, closed_store_ids, outfile='output.csv'):
    """
    Trains a random forest for each store and generates predictions for all testing input
    :param train_df: processed dataframe of training data
    :param test_df: processed dataframe of testing data (should not have Sales column)
    :param closed_store_ids: list of stores that are closed in the testing data
    :param outfile: name of the output file to write predictions to
    :return:
    """
    train_stores = dict(list(train_df.groupby('Store')))
    test_stores = dict(list(test_df.groupby('Store')))
    open_store_sales = pd.Series()
    train_scores = []
    for i in test_stores:
        # current store
        current_store = train_stores[i]

        # define training and testing sets
        train_x = current_store.drop(['Id', 'Sales', 'Store', 'Open'], axis=1)
        train_y = current_store['Sales']
        # print(X_train)
        # print(Y_train)

        test_x = test_stores[i].copy()
        test_store_ids = test_x['Id']
        test_x = test_x.drop(['Id', 'Store'], axis=1)
        # X_test['Customers'] = np.log2(X_test['Customers'] + 1)
        # X_test.drop(['Customers'], axis=1, inplace=True)
        # print(X_test)

        model = LinearRegression()
        model.fit(train_x, train_y)
        test_y = model.predict(test_x)
        train_scores.append(model.score(train_x, train_y))

        # append predicted values of current store to submission
        open_store_sales = open_store_sales.append(pd.Series(test_y, index=test_store_ids))
        print('Completed Store %d: train_score=%.5f' % (i, train_scores[-1]))

    # save to csv file
    open_store_sales = pd.DataFrame(
        {'Id': open_store_sales.index + 1, 'Sales': np.power(2, open_store_sales.values)})
    closed_store_sales = pd.DataFrame(
        {'Id': closed_store_ids + 1, 'Sales': 1})  # 0 sales need to be map to 1 for kaggle

    submission = pd.concat([open_store_sales, closed_store_sales])
    submission.to_csv(outfile, index=False)
    print('done: wrote predictions to %s' % outfile)


train_models(train, test, closed_store_ids, outfile='rossmann-rf-per-store.csv')
