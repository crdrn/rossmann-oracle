from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import settings

import argparse
import pandas as pd
import sys
import tempfile
import tensorflow as tf

# Boolean features
is_open = tf.feature_column.numeric_column('Open')
has_day_promo = tf.feature_column.numeric_column('Promo')
school_holiday = tf.feature_column.numeric_column('SchoolHoliday')

# Continuous numeric base features
sales = tf.feature_column.numeric_column('Sales')
customers = tf.feature_column.numeric_column('Customers')

# Date-related features -- use bucketization or convert ?
state_holiday = tf.feature_column.categorical_column_with_vocabulary_list(
    'StateHoliday', vocabulary_list=['0', 'a', 'c'])
day = tf.feature_column.numeric_column('DayOfWeek')

"""
promo_interval = tf.feature_column.categorical_column_with_hash_bucket(
    'PromoInterval', hash_bucket_size=1000)
promo_week = tf.feature_column.categorical_column_with_hash_bucket(
    'Promo2SinceWeek', hash_bucket_size=1000)
promo_year = tf.feature_column.categorical_column_with_hash_bucket(
    'Promo2SinceYear', hash_bucket_size=1000)
has_store_promo = tf.feature_column.numeric_column('Promo2')
"""

# Categorical features
store_type = tf.feature_column.categorical_column_with_vocabulary_list(
    'StoreType', vocabulary_list=['c', 'a', 'd', 'b'])
assortment = tf.feature_column.categorical_column_with_vocabulary_list(
    'Assortment', vocabulary_list=['a' 'c' 'b'])

is_in_competition = tf.feature_column.numeric_column('IsInCompetition')

base_columns = [is_open, has_day_promo, school_holiday, state_holiday, day, customers, store_type, assortment,
                is_in_competition]

crossed_columns = []

deep_columns = [tf.feature_column.indicator_column(is_open),
                tf.feature_column.indicator_column(has_day_promo),
                tf.feature_column.indicator_column(school_holiday),
                tf.feature_column.indicator_column(state_holiday),
                tf.feature_column.indicator_column(day),
                customers,
                tf.feature_column.indicator_column(store_type),
                tf.feature_column.indicator_column(assortment),
                tf.feature_column.indicator_column(is_in_competition),
                ]


def build_estimator(model_dir, model_type):
    """Build an estimator."""
    if model_type is 'wide':
        m = tf.estimator.LinearRegressor(
            model_dir=model_dir,
            feature_columns=base_columns + crossed_columns)
    elif model_type is 'deep':
        m = tf.estimator.DNNRegressor(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 50])
    else:
        m = tf.estimator.DNNLinearCombinedRegressor(
            model_dir=model_dir,
            linear_feature_columns=crossed_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50])
    return m


def input_fn_train(data_file, num_epochs, shuffle):
    """Input builder function."""
    df = pd.read_csv(data_file, skipinitialspace=False, low_memory=False)

    # get only columns of interest
    df = df[settings.TRAIN_FEATURES]

    # remove NaN elements
    df = df.dropna(how='any', axis=0)

    # extract labels
    y_values = df['Sales']

    return tf.estimator.inputs.pandas_input_fn(
        x=df.drop('Sales', axis=1),
        y=y_values,
        target_column='Sales',
        batch_size=100,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=1)


def input_fn_test(data_file, num_epochs, shuffle):
    """Input builder function."""
    df = pd.read_csv(data_file, skipinitialspace=False, low_memory=False)

    # get only columns of interest
    df = df[settings.TEST_FEATURES]

    # fill in NAN competition distances
    df['CompetitionDistance'].fillna(9999, inplace=True)

    # remove rows with NaN elements
    df = df.dropna(how='any', axis=0)

    df['StateHoliday'] = df['StateHoliday'].astype(str)

    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        num_epochs=num_epochs,
        target_column='Sales',
        shuffle=shuffle,
        num_threads=1)


def train_model(model_dir, model_type, train_steps, train_file_name):
    """Train and evaluate the model."""
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir

    m = build_estimator(model_dir, model_type)

    # set num_epochs to None to get infinite stream of data.
    m.train(input_fn=input_fn_train(train_file_name, num_epochs=None, shuffle=True),
            steps=train_steps)

    return m


def format_predictions(predictions, outfile='output.csv'):
    results = [p['predictions'][0] for p in list(predictions)]
    df = pd.DataFrame(results, columns=['Sales'])

    # +1 to the id so we start from 1 instead of 0
    df.index += 1

    df.to_csv(outfile, index_label='Id')
    return


def main(_):
    print('Training %s model on\ntrain = %s\ntest = %s\nfor %s steps' % (FLAGS.model_type, FLAGS.train_data, FLAGS.test_data,
                                                       FLAGS.train_steps))
    # train da model
    model = train_model(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps, FLAGS.train_data)
    print(model.get_variable_names())

    # try predicting
    predictions = model.predict(input_fn=input_fn_test(settings.CSV_TREATED_TEST, num_epochs=1, shuffle=False))
    format_predictions(predictions, FLAGS.output)
    return


FLAGS = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Base directory for output models."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="wide",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=5000,
        help="Number of training steps."
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=settings.CSV_TREATED_TRAIN,
        help="Path to the training data."
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=settings.CSV_TREATED_TEST,
        help="Path to the test data."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.csv",
        help="Path to write results to."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
