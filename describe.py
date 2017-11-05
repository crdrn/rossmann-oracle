#!/usr/bin/python
import pandas as pd
from settings import CSV_SMALL_TRAIN, CSV_TEST, CSV_STORE, TRAIN_FEATURES

# Load the data into a DataFrame
train = pd.read_csv(CSV_SMALL_TRAIN, low_memory=False)
train = train[TRAIN_FEATURES]
test = pd.read_csv(CSV_TEST, low_memory=False)
store = pd.read_csv(CSV_STORE, low_memory=False)

def summarize(df):
    print(train.describe())
    print('\n%13s %15s %s' % ('Column', 'Type', 'Unique Values'))
    print('-----------------------------------------------------')
    for col in df:
        print('%13s %15s %s' % (col, df[col].dtype, df[col].unique()))
    return


print(CSV_SMALL_TRAIN)
x = train.groupby('Store')
print(x.describe())