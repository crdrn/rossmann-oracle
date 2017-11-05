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
TRAIN_FEATURES = ['DayOfWeek', 'Date', 'Store', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday',
                  'SchoolHoliday', 'StoreType','Assortment', 'CompetitionDistance', 'IsInCompetition', 'Promo2',
                  'Median']
TRAIN_FEATURES = ['Date','Store','Sales','Customers','Open', 'Promo','Assortment','Promo2']
TEST_FEATURES = list(TRAIN_FEATURES)
TEST_FEATURES.remove('Sales')
ADDED_FEATURES = ['Median','DaysTillChristmas']
TEST_FEATURES += ADDED_FEATURES
ALL_FEATURES=['Store','DayOfWeek','Date','Sales','Customers','Open','Promo','StateHoliday',
              'SchoolHoliday','StoreType','Assortment','CompetitionDistance',
              'CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2','Promo2SinceWeek',
              'Promo2SinceYear','PromoInterval','DateOfCompetitionOpen','IsInCompetition']