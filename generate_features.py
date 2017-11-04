#!/usr/bin/python
import pandas as pd
import settings
from describe import summarize

# Load the data into a DataFrame
train = pd.read_csv(settings.CSV_TRAIN, low_memory=False)
test = pd.read_csv(settings.CSV_TEST, low_memory=False)
store = pd.read_csv(settings.CSV_STORE, low_memory=False)

# merge store.csv into the train and test data
train_store = pd.merge(train, store, on='Store', how='outer')
test_store = pd.merge(test, store, on='Store', how='outer')

def one_hot_encode(df):
    # one hot encode categorical variables
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df['DayOfWeek'] = df['DayOfWeek'].replace([1, 2, 3, 4, 5, 6, 7], ['day_mon', 'day_tue', 'day_wed', 'day_thur',
                                                                      'day_fri', 'day_sat', 'day_sun'])
    df['SchoolHoliday'] = df['SchoolHoliday'].replace([0, 1], ['sch_hol_none', 'sch_hol_yes'])
    df['Open'] = df['Open'].replace([0, 1], ['open_no', 'open_yes'])
    df['Promo'] = df['Promo'].replace([0, 1], ['day_promo_none', 'day_promo_yes'])
    df['StateHoliday'] = df['StateHoliday'].replace(['0', 'a', 'c'], ['state_hol_none', 'state_hol_public',
                                                                      'state_hol_christmas'])
    df['StoreType'] = df['StoreType'].replace(['a', 'b', 'c', 'd'], ['store_type_a', 'store_type_b', 'store_type_c',
                                                                     'store_type_d'])
    df['Assortment'] = df['Assortment'].replace(['a', 'b', 'c'], ['assortment_a', 'assortment_b', 'assortment_c'])
    df['IsInCompetition'] = df['IsInCompetition'].replace([0, 1], ['in_competition', 'not_in_competition'])
    df['Promo2'] = df['Promo2'].replace([0, 1], ['promo2_none', 'promo2_yes'])

    for var in settings.CATEGORICAL_FEATURES:
        one_hot = pd.get_dummies(df[var])
        df = df.drop(var, axis=1)
        df = df.join(one_hot)
    return df

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
train_store.to_csv(settings.CSV_TREATED_TRAIN, sep=',', index=False)
test_store.to_csv(settings.CSV_TREATED_TEST, sep=',', index=False)
