import pandas as pd
from settings import *

big = pd.read_csv(CSV_TREATED_TRAIN, low_memory=False)
train=big.sample(frac=0.8,random_state=200)
left=big.drop(train.index)
test=left.sample(frac=0.5, random_state=874)
dev=left.drop(test.index)

train.to_csv(CSV_SMALL_TRAIN, index=False)
test.to_csv(CSV_SMALL_TEST, index=False)
dev.to_csv(CSV_SMALL_DEV, index=False)