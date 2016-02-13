#!/bin/bash
vw -d train.txt --passes 200 --ngram 2 -k -c -f model.vw --loss_function quantile -l 600 -q tc -b 27 --initial_t 1 --power_t 0.3
vw -d test.txt -i model.vw -t -p predictions.txt
python3 parsPred.py predictions.txt data/features-test.csv # 1st - prediction, 2nd - control set (for IDs)