import settings
import pandas as pd

def evaluate_predictions(predictions, actual):
    """ Returns RMSPE given lists of predictions and corresponding actual sales """
    sum = 0
    for estimate, real in zip(predictions, actual):
        real = 1 if real==0 else real
        sum += pow((float(real) - estimate)/real, 2)
    return pow(sum/len(actual), 0.5)

def evaluate(prediction_csv):
    answers = pd.read_csv(settings.CSV_SMALL_TEST)['Sales']
    prediction = pd.read_csv(prediction_csv)['Sales']
    print("RMSPE: {}".format(evaluate_predictions(prediction, answers)))


def main():
    prediction_file = 'output.csv'
    print("Evaluating RMSPE of '{}' compared to '{}'".format(prediction_file, settings.CSV_SMALL_TEST))
    evaluate(prediction_file)

if __name__ == '__main__':
    main()