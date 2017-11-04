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

SEED = 3244
DIM_INPUT = len(settings.TRAIN_FEATURES) - 1

# fix random seed for reproducibility
np.random.seed(SEED)

def preprocess(df):
    df['CompetitionDistance'].fillna(9999, inplace=True)

    # recode StateHoliday
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df['StateHoliday'] = df['StateHoliday'].replace(['0', 'a', 'c'], [0, 1, -1])
    df['StoreType'] = df['StoreType'].replace(['a', 'b', 'c', 'd'], [1, 2, 3, 4])
    df['Assortment'] = df['Assortment'].replace(['a', 'b', 'c'], [1, 2, 3])
    return df


# load train data
df_train = pd.read_csv(settings.CSV_TREATED_TRAIN, low_memory=False)
df_train = df_train[settings.TRAIN_FEATURES]
df_train = preprocess(df_train)
train_y = np.array(df_train['Sales'].values)
train_y = train_y.reshape(-1, 1)
train_x = np.array(df_train.drop('Sales', axis=1).values)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)

df_test = pd.read_csv(settings.CSV_TREATED_TEST, low_memory=False)
df_test = df_test[settings.TEST_FEATURES]
df_test = preprocess(df_test)
test_x = np.array(df_test.values)
test_x.reshape(-1, 1)

test_x = scaler.transform(test_x)

print(train_x)
print(train_y)
print(test_x)

# define base model
def make_model():
    # create model
    """
    model = Sequential()
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    """
    model = Sequential()
    model.add(Dense(DIM_INPUT, input_dim=DIM_INPUT, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal'))
    model.add(Dense(1000, kernel_initializer='normal'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model


model = make_model()
sales = model.fit(train_x, train_y, epochs=5, batch_size=100, verbose=2, shuffle=True)

predicted_sales = model.predict(test_x)
#inv_predicted_sales = scaler.inverse_transform(predicted_sales)

def write_to_submission_csv(predictions, outfile):
    with open(outfile, 'w') as submissionfile:
        i = 1
        # print header
        HEADER = '\"Id\",\"Sales\"\n'
        print(HEADER)
        submissionfile.write(HEADER)
        for p in predictions:
            line = '%d,%f\n' % (i, p)
            print(line)
            submissionfile.write(line)
            i += 1

    return

write_to_submission_csv(predicted_sales, 'out-keras.csv')
