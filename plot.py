import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

datfile = "parsed_results.csv"

data = pd.read_csv(datfile)

print(data)

xvals = ['rows', 'cols', 'nnz']
yvals = data.columns[6:]
print(yvals)

sns.set_theme()

for xval in xvals:
  for yval in yvals:

    plot = data.plot.scatter(x=xval, y=yval)
    plot.set_xscale('log')

    plt.savefig("plots/" + xval + "_vs_" + yval + ".png")
    plt.close()