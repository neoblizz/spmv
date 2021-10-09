import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 3:
  print("Usage: python3 ./plot/py parsed_datafile.csv plotdir")
  sys.exit(1)

datafile = sys.argv[1]
plotdir = sys.argv[2]

os.mkdir(plotdir)

data = pd.read_csv(datafile)

print(data)

xvals = ['rows', 'cols', 'nnz']
yvals = data.columns[6:]
print(yvals)

sns.set_theme()

for xval in xvals:
  for yval in yvals:

    plot = data.plot.scatter(x=xval, y=yval)
    plot.set_xscale('log')
    plot.set_yscale('log')

    plt.savefig(plotdir + "/" + xval + "_vs_" + yval + ".png")
    plt.close()
