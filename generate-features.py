#!/usr/bin/python
import pandas as pd
from settings import CSV_TRAIN, CSV_TEST, CSV_STORE, CSV_TREATED_TRAIN, CSV_TREATED_TEST
from describe import summarize

# Load the data into a DataFrame
train = pd.read_csv(CSV_TRAIN, low_memory=False)
test = pd.read_csv(CSV_TEST, low_memory=False)
store = pd.read_csv(CSV_STORE, low_memory=False)

# merge store.csv into the train and test data
train_store = pd.merge(train, store, on='Store', how='outer')
test_store = pd.merge(test, store, on='Store', how='outer')


def generate_is_in_competition(df):
    # construct datestamp YYYY-MM-DD for date of competition's opening
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateOfCompetitionOpen'] = pd.to_datetime(df.CompetitionOpenSinceYear*10000 + df.CompetitionOpenSinceMonth*100 + 15,format='%Y%m%d')

    # now make the flag
    df['IsInCompetition'] = (df['DateOfCompetitionOpen'] <= df['Date']) * 1
    return df

train_store = generate_is_in_competition(train_store)
test_store = generate_is_in_competition(test_store)

#summarize(train_store)
#print(train_store)

# save dataframes to csv
train_store.to_csv(CSV_TREATED_TRAIN, sep=',', index=False)
test_store.to_csv(CSV_TREATED_TEST, sep=',', index=False)
