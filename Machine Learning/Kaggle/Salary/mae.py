import pandas as pd
import numpy as np
import sys

target = pd.read_csv(sys.argv[2], header=None)
pred = pd.read_csv(sys.argv[1], header=None)
print('Model MAE:', np.mean(
    np.abs(target.iloc[:, 0].values - pred.iloc[:, 0].values)))
