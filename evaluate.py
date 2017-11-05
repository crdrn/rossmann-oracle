import settings
import pandas

def evaluate_predictions(predictions, actual):
    """ Returns RMSPE given lists of predictions and corresponding actual sales """
    sum = 0
    for estimate, real in zip(predictions, actual):
        real = 1 if real==0 else real
        sum += pow((float(real) - estimate)/real, 2)
    return pow(sum/len(actual), 0.5)

def