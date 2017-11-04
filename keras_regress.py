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
import generate_features

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

    df = generate_features.one_hot_encode(df)
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
    model.add(Dense(500, kernel_initializer='normal'))
    model.add(Dense(50, kernel_initializer='normal'))
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
