from scipy.stats import wilcoxon
# d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]
#
# w, p = wilcoxon(d)
#
# print("w: {}, p: {}".format(w, p))


import pandas as pd
df = pd.read_csv("blood_pressure.csv")
a = df[['bp_before','bp_after']].describe()
print(a)

from scipy import stats
b = stats.shapiro(df['bp_before'])
print(b)

rel = stats.ttest_rel(df['bp_before'], df['bp_after'])
print('rel: ', rel)


wilcoxon = stats.wilcoxon(df['bp_before'], df['bp_after'])

print('wilcoxon: ', wilcoxon)

for i in range(0,100,5):
  print(i)