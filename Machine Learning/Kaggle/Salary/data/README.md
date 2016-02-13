download data on https://inclass.kaggle.com/c/cmc-msu-machine-learning-spring-14-15-competition-1

train: default
train2: classified title, top 50 words
train3: classified title, all top words
train4: default index, top words
train5: full loc
train6: namespaces with no name
train7: one namespace for all
train_base: all features with namespaces
train_log: np.log(salary)
train8: all features with namespaces, without locRaw

-- tmp: train_base --
6805 - std
6795 - --initial_t 0.5
6647 - no skips
6592 - --cubic deg
6448 - --power_t 0.3 --l1 0.00000005
5923 - -b 27
vw -d data/train_base.bu.txt --passes 100 --ngram 2 -k -c -f model.vw --loss_function quantile -l 1500 --initial_t 1 --cubic deg --power_t 0.3 --l1 0.00000005 -b 27

-- tmp: train --
5905 - vw -d data/train.bu.txt --passes 100 --ngram 2 -k -c -f model.vw --loss_function quantile -l 1000 -q tc -b 27 --initial_t 1 --power_t 0.3

6425 - no -b 27
6420 - -l 800
6412 - -l 500
vw -d train.txt --passes 100 --ngram 2 -k -c -f model.vw --loss_function quantile -l 500 -q tc --initial_t 1 --power_t 0.3 -b 27
