import numpy as np
import pandas as pd
import settings
import scipy
from sklearn.ensemble import RandomForestRegressor

def format_output(results, datacsv, outfile="output.csv"):
    # Convert all predictions when store is closed to 1
    is_open = pd.read_csv(datacsv)['Open']
    df = pd.DataFrame(np.exp(results), columns=['Sales'])
    df.where(is_open == 1, 1, inplace=True)
    df.where(df != 0, 1, inplace=True)

    # +1 to the id so we start from 1 instead of 0
    df.index += 1

    df.to_csv(outfile, index_label='Id')
    return

def fix_datatypes(df):
    #  fix missing and non-numeric data

    holidays = pd.get_dummies(df['StateHoliday'])
    if 'a' in df.columns:
        df.loc[:,'Holiday'] = holidays['0'] + holidays['a']
    elif '0' in df.columns:
        df.loc[:,'Holiday'] = holidays['0']
    else:
        df.loc[:,'Holidays'] = 0

    df.loc[:,'Christmas'] = holidays['c'] if 'c' in df.columns else 0

    df.loc[:,'CompetitionDistance'] = np.log(df['CompetitionDistance'].fillna(99999))

    assortment = pd.get_dummies(df['Assortment'])
    df.loc[:,'AssortmentA'] = assortment['a'] if 'a' in assortment else 0
    df.loc[:,'AssortmentB'] = assortment['b'] if 'b' in assortment else 0
    df.loc[:,'AssortmentC'] = assortment['c'] if 'c' in assortment else 0

    storetype = pd.get_dummies(df['StoreType'])
    df.loc[:,'storetypeA'] = storetype['a'] if 'a' in storetype else 0
    df.loc[:,'storetypeB'] = storetype['b'] if 'b' in storetype else 0
    df.loc[:,'storetypeC'] = storetype['c'] if 'c' in storetype else 0
    df.loc[:,'storetypeD'] = storetype['d'] if 'd' in storetype else 0

    date = pd.to_datetime(df['Date'])
    df.loc[:,'Day'] = date.map(lambda x: x.month)

    df.drop(labels=['StateHoliday', 'Assortment', 'StoreType', 'Date'], axis=1, inplace=True)

    return df


train = pd.read_csv(settings.CSV_SMALL_TRAIN)
X = train[settings.TEST_FEATURES]
y = np.log(train['Sales']+1)

X = fix_datatypes(X)

forest = RandomForestRegressor(n_estimators=10)
forest.fit(X,y)

test = pd.read_csv(settings.CSV_TREATED_TEST)
testX = test[settings.TEST_FEATURES]
testX = fix_datatypes(testX)
format_output(forest.predict(testX), settings.CSV_TREATED_TEST)