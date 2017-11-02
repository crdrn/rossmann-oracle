from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from settings import *

import argparse
import pandas as pd
import sys
import tempfile
import tensorflow as tf

LABELS = ['Store','DayOfWeek','Date','Sales','Customers','Open','Promo','StateHoliday','SchoolHoliday','StoreType',
          'Assortment','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2',
          'Promo2SinceWeek','Promo2SinceYear','PromoInterval','DateOfCompetitionOpen','IsInCompetition']

# Boolean features
is_open = tf.feature_column.numeric_column("Open")
has_day_promo = tf.feature_column.numeric_column("Promo")
school_holiday = tf.feature_column.numeric_column("SchoolHoliday")

# Continuous numeric base features
sales = tf.feature_column.numeric_column("Sales")
customers = tf.feature_column.numeric_column("Customers")

# Date-related features -- use bucketization or convert ?
state_holiday = tf.feature_column.categorical_column_with_vocabulary_list(
    "StateHoliday", vocabulary_list=["0", "a", "c"])
day = tf.feature_column.numeric_column("DayOfWeek")

promo_interval = tf.feature_column.categorical_column_with_hash_bucket(
    "PromoInterval", hash_bucket_size=1000)
promo_week = tf.feature_column.categorical_column_with_hash_bucket(
    "Promo2SinceWeek", hash_bucket_size=1000)
promo_year = tf.feature_column.categorical_column_with_hash_bucket(
    "Promo2SinceYear", hash_bucket_size=1000)
has_store_promo = tf.feature_column.numeric_column("Promo2")

# Categorical features
store_type = tf.feature_column.categorical_column_with_vocabulary_list(
    "StoreType", vocabulary_list=['c', 'a', 'd', 'b'])
assortment = tf.feature_column.categorical_column_with_vocabulary_list(
    "Assortment", vocabulary_list=['a' 'c' 'b'])

base_columns = [is_open, has_day_promo, school_holiday, state_holiday, day, customers]
crossed_columns = []
deep_columns = []


def build_estimator(model_dir, model_type):
    """Build an estimator."""
    if model_type == "wide":
        m = tf.estimator.LinearRegressor(
            model_dir=model_dir,
            feature_columns=base_columns + crossed_columns)
    elif model_type == "deep":
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


def input_fn(data_file, num_epochs, shuffle):
    """Input builder function."""
    df_data = pd.read_csv(
        tf.gfile.Open(data_file),
        skipinitialspace=False,
        names=LABELS,
        engine="python",
        skiprows=1)
    # remove NaN elements
    df_data = df_data.dropna(how="any", axis=0)
    y_values = df_data['Sales']

    return tf.estimator.inputs.pandas_input_fn(
        x=df_data,
        y=y_values,
        batch_size=100,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=1)


def train_and_eval(model_dir, model_type, train_steps, train_file_name, test_file_name):
    """Train and evaluate the model."""
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir

    m = build_estimator(model_dir, model_type)
    # set num_epochs to None to get infinite stream of data.
    m.train(input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
            steps=train_steps)
    # set steps to None to run evaluation until all data consumed.
    print("TRAINING DONE")
    results = m.evaluate(input_fn=
                         input_fn(test_file_name, num_epochs=1, shuffle=False), steps=None)
    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


def main(_):
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                   FLAGS.train_data, FLAGS.test_data)


FLAGS = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Base directory for output models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wide",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=5000,
        help="Number of training steps."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=CSV_SMALL_TRAIN,
        help="Path to the training data."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=CSV_SMALL_TEST,
        help="Path to the test data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
