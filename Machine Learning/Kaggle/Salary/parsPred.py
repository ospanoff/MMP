import numpy as np
import pandas as pd
import sys

if len(sys.argv) < 2:
    sys.exit("Usage: python3 parsPred.py predName testData")

# Pars prediction file
pred = pd.read_csv(sys.argv[1], header=None)
test = pd.read_csv(sys.argv[2])

outPre = open("predictions.csv", 'w')
outPre.write("Id,Prediction\n")
i = 0
for val in test.values[:, 0].astype(np.str):
    outPre.write(val + ',' + pred.values[i, 0].astype(np.str) + '\n')
    i += 1
outPre.close()
