# import pandas as pd
# input = pd.read_csv('./human-evaluation-wan.csv')
#
# input.iloc[0]
# print(input)

import csv
import json
import itertools
import numpy as np
from scipy import stats
import sys

USERS = ['wan', 'wan2', 'wan3', 'wan4']


shuffle = json.load(open('all_models_info.json', 'r'))
print(shuffle)
orders = shuffle['0']

scores = {'human': [], 'alamo': [], 'code2seq': [], 'deepcom': [], 'seq2seq': [], 'hybrida2c': []}

rows = {user: [] for user in USERS}
for user in USERS:
    with open('human-evaluation-{}.csv'.format(user)) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            print(row, '==> ', len(row))
            rows[user].append(row)
        rows[user] = rows[user][1:]

    assert len(rows[user]) == 200

for i in range(100):
    for j, order in enumerate(['human'] + orders[i]):
        print('2*i+1: ', 2*i+1)
        print("rows[user][2*i+1][1:]: ", rows[user][2*i+1][1:])
        raw_scores = []
        for user in USERS:
            raw_score = [int(i) for i in rows[user][2*i+1][1:]]
            # if 0 in raw_score:
            #     print('iiiiii: ', i)
            #     sys.exit()
            raw_scores.append(raw_score[j])
        scores[order].append(raw_scores)

print('scores: ', scores)

scores_onelist = {'human': [], 'alamo': [], 'code2seq': [], 'deepcom': [], 'seq2seq': [], 'hybrida2c': []}
for key in scores_onelist.keys():
    scores_onelist[key] = list(itertools.chain.from_iterable(scores[key]))

scores_avg_onelist = {'human': [], 'alamo': [], 'code2seq': [], 'deepcom': [], 'seq2seq': [], 'hybrida2c': []}
for key, value in scores.items():
    scores_avg_onelist[key] = [np.mean(l) for l in value]
print('scores_avg_onelist: ', scores_avg_onelist)

scores_statistic = {'human': None, 'alamo': None, 'code2seq': None, 'deepcom': None, 'seq2seq': None, 'hybrida2c': None}
for key, value in scores_onelist.items():
    statistic = dict((l, value.count(l)) for l in set(value))
    statistic['avg'] = np.mean(value)
    statistic['ge2'] = len(list(filter(lambda x: x >= 2, value)))
    statistic['ge3'] = len(list(filter(lambda x: x >= 3, value)))
    statistic['ge4'] = len(list(filter(lambda x: x >= 4, value)))
    scores_statistic[key] = statistic
    # scores_statistic[key]['avg'] = 2
print('==>scores_statistic: ', scores_statistic)

rel = stats.ttest_rel(scores_avg_onelist['alamo'], scores_avg_onelist['code2seq'])
print('==>rel: ', rel)

wilcoxon = stats.wilcoxon(scores_avg_onelist['alamo'], scores_avg_onelist['code2seq'])
wilcoxon2 = stats.wilcoxon(scores_avg_onelist['alamo'], scores_avg_onelist['hybrida2c'])

print('==>wilcoxon: ', wilcoxon)
print('==>wilcoxon2: ', wilcoxon2)




