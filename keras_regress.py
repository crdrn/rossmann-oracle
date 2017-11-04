import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import settings
import sys

SEED = 3244
DIM_INPUT = 29

# fix random seed for reproducibility
np.random.seed(SEED)

def preprocess(df):
    df['CompetitionDistance'].fillna(999999, inplace=True)

    # set 0 sales to 1
    #df['Sales'] = df['Sales'] +1
    # set closed days to sale of 1
    #df.loc[df['Open'] == 0, 'Sales'] = 1
    df['Customers'] = np.log(df['Customers'] + 1)
    df['CompetitionDistance'] = np.log(df['CompetitionDistance'] + 1)

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


# load train data
df_train = pd.read_csv(settings.CSV_TREATED_TRAIN, low_memory=False)
df_train = df_train[settings.TRAIN_FEATURES]
df_train = preprocess(df_train)
print(df_train)
print(df_train.dtypes)
train_y = np.array(df_train['Sales'].values)
train_y = train_y.reshape(-1, 1)
train_y = np.log(train_y + 1)

train_x = np.array(df_train.drop('Sales', axis=1).values)

#scaler = StandardScaler()
#train_x = scaler.fit_transform(train_x)

df_test = pd.read_csv(settings.CSV_TREATED_TEST, low_memory=False)
df_test = df_test[settings.TEST_FEATURES]
df_test = preprocess(df_test)
df_test['state_hol_public'] = 0
df_test['state_hol_christmas'] = 0
print(df_test.dtypes)
test_x = np.array(df_test.values)
test_x.reshape(-1, 1)

#test_x = scaler.transform(test_x)

print(train_x)
print(train_y)
print(test_x)

# define base model
def make_model():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=DIM_INPUT, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal'))
    model.add(Dense(50, kernel_initializer='normal'))
    model.add(Dense(25, kernel_initializer='normal'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


model = make_model()
sales = model.fit(train_x, train_y, epochs=50, batch_size=256, verbose=2, shuffle=True)

predicted_sales = model.predict(test_x)
#inv_predicted_sales = scaler.inverse_transform(predicted_sales)

def write_to_submission_csv(predictions, outfile):
    with open(outfile, 'w') as submissionfile:
        i = 1
        # print header
        HEADER = '\"Id\",\"Sales\"\n'
        print(HEADER.rstrip())
        submissionfile.write(HEADER)
        for p in predictions:
            line = '%d,%f\n' % (i, np.exp(p))
            print(line.rstrip())
            submissionfile.write(line)
            i += 1

    return

write_to_submission_csv(predicted_sales, 'out-keras.csv')
