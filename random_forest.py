import numpy as np
import pandas as pd
import settings
from sklearn.ensemble import RandomForestRegressor


def format_output(results, datacsv, outfile="output.csv"):
    # Convert all predictions when store is closed to 1
    is_open = pd.read_csv(datacsv)['Open']
    df = pd.DataFrame(pow(2, (results)), columns=['Sales'])
    df.where(is_open == 1, 1, inplace=True)
    df.where(df != 0, 1, inplace=True)

    # +1 to the id so we start from 1 instead of 0
    df.index += 1

    df.to_csv(outfile, index_label='Id')
    return


def fix_datatypes(df):
    #  fix missing and non-numeric data

    if 'StateHoliday' in df:
        df = format_stateHoliday(df)

    if 'CompetitionDistance' in df:
        df.loc[:, 'CompetitionDistance'] = np.log2(df['CompetitionDistance'].fillna(9999999))
        df.drop(labels=['CompetitionDistance'], axis=1, inplace=True)

    if 'Assortment' in df:
        assortment = pd.get_dummies(df['Assortment'])
        df.loc[:, 'AssortmentA'] = assortment['a'] if 'a' in assortment else 0
        df.loc[:, 'AssortmentB'] = assortment['b'] if 'b' in assortment else 0
        df.loc[:, 'AssortmentC'] = assortment['c'] if 'c' in assortment else 0
        df.drop(labels=['Assortment'], axis=1, inplace=True)

    if 'StoreType' in df:
        storetype = pd.get_dummies(df['StoreType'])
        df.loc[:, 'storetypeA'] = storetype['a'] if 'a' in storetype else 0
        df.loc[:, 'storetypeB'] = storetype['b'] if 'b' in storetype else 0
        df.loc[:, 'storetypeC'] = storetype['c'] if 'c' in storetype else 0
        df.loc[:, 'storetypeD'] = storetype['d'] if 'd' in storetype else 0
        df.drop(labels=['StoreType'], axis=1, inplace=True)

    if 'Date' in df:
        date = pd.to_datetime(df['Date'])
        df.loc[:, 'Day'] = date.map(lambda x: x.month)
        df.drop(labels=['Date'], axis=1, inplace=True)

    return df


def format_stateHoliday(df):
    holidays = pd.get_dummies(df['StateHoliday'])
    if 'a' in df.columns:
        df.loc[:, 'Holiday'] = holidays['0'] + holidays['a']
    elif '0' in df.columns:
        df.loc[:, 'Holiday'] = holidays['0']
    else:
        df.loc[:, 'Holidays'] = 0
    df.loc[:, 'Christmas'] = holidays['c'] if 'c' in df.columns else 0

    df.drop(labels=['StateHoliday'], axis=1, inplace=True)
    return df


def add_median(df, median):
    return pd.merge(df, median, on='Store', how='outer')


def calculate_median(df):
    median = pd.DataFrame(df.groupby(['Store'], as_index=False).median()[['Store', 'Sales']])
    median.columns = ['Store', 'Median']
    median['Median'] = np.log2(median['Median'])
    return median


def add_to_christmas(df):
    today = pd.to_datetime(df['Date'])
    christmas = pd.to_datetime("25/12/2015")
    df['DaysTillChristmas'] = (today - christmas).apply(lambda x: x.days % 182)
    return df


def get_features(df, median):
    ''' Gets features to be used in the training/testing '''
    df = add_median(df, median)
    df = add_to_christmas(df)
    df = df[settings.TEST_FEATURES]
    df = fix_datatypes(df)
    return df


def main():
    train = pd.read_csv(settings.CSV_SMALL_TRAIN)

    # only train on examples where sales != 0
    train = train[train['Sales'] != 0]

    median = calculate_median(train)
    X = get_features(train, median)

    # log2 the sales score
    y = np.log(train['Sales'] + 1)

    forest = RandomForestRegressor(n_estimators=100, max_depth=10, oob_score=True)
    forest.fit(X, y)

    # test RandomForest
    test = pd.read_csv(settings.CSV_TREATED_TEST)
    testX = get_features(test, median)
    format_output(forest.predict(testX), settings.CSV_TREATED_TEST)
    return


if __name__ == '__main__':
    main()
