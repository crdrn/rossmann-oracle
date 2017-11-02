#!/usr/bin/python
import pandas as pd
from settings import CSV_TRAIN, CSV_TEST, CSV_STORE

# Load the data into a DataFrame
train = pd.read_csv(CSV_TRAIN, low_memory=False)
test = pd.read_csv(CSV_TEST, low_memory=False)
store = pd.read_csv(CSV_STORE, low_memory=False)

# merge store.csv into the train and test data
train_store = pd.merge(train, store, on='Store', how='outer')
test_store = pd.merge(test, store, on='Store', how='outer')

#print(train_store)
