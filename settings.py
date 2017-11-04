#!/usr/bin/python

CSV_TRAIN = 'data/train_v2.csv'
CSV_TEST = 'data/test_v2.csv'
CSV_STORE = 'data/store.csv'

CSV_TREATED_TRAIN = 'data/treated/train.csv'
CSV_TREATED_TEST = 'data/treated/test.csv'

CSV_SMALL_TRAIN = 'data/treated/small_train.csv'
CSV_SMALL_TEST = 'data/treated/small_test.csv'
CSV_SMALL_DEV = 'data/treated/small_dev.csv'

CATEGORICAL_FEATURES = ['DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType',
                        'Assortment', 'IsInCompetition', 'Promo2']
TRAIN_FEATURES = ['DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday',
                  'SchoolHoliday', 'StoreType','Assortment', 'CompetitionDistance', 'IsInCompetition', 'Promo2']
TEST_FEATURES = list(TRAIN_FEATURES)
TEST_FEATURES.remove('Sales')
