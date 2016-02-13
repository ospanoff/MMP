#!/bin/bash
head -n 100000 data/train.bu.txt > train.txt
vw -d train.txt --passes 100 --ngram 2 -k -c -f model.vw --loss_function quantile -l 600 -q tc -b 27 --initial_t 1 --power_t 0.3
# vw -d train.txt --passes 100 --ngram 2 -k -c -f model.vw --loss_function quantile -l 500 -q tc --initial_t 1 --power_t 0.3 -b 28
# vw -d train.txt --passes 100 --ngram t5 --ngram 2 -k -c -f model.vw --loss_function quantile -l 1000 -q tc --initial_t 1 --power_t 0.3
# head -n 50000 train.txt > test.txt
tail -n 60000 data/train.bu.txt > test.txt
vw -d test.txt -i model.vw -t -p predictions.txt
cut -d' ' -f1 test.txt > answers.txt
python3 mae.py predictions.txt answers.txt