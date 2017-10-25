#!/usr/bin/python
import pandas as pd

CSV_TRAIN = 'data/train_v2.csv'
CSV_TEST = 'data/test_v2.csv'

# Load the data into a DataFrame
train = pd.read_csv(CSV_TRAIN, low_memory=False)
test = pd.read_csv(CSV_TEST, low_memory=False)

def summarize(df):
    print(train.describe())
    print('\n%13s %15s %s' % ('Column', 'Type', 'Unique Values'))
    print('-----------------------------------------------------')
    for col in df:
        print('%13s %15s %s' % (col, df[col].dtype, df[col].unique()))
    return


print(CSV_TRAIN)
summarize(train)

print('\n\n')
print(CSV_TEST)
summarize(test)
