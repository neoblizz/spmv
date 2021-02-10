import pandas as pd 
import numpy as np 
import altair as alt
from altair_saver import save
import matplotlib.pyplot as plt

data = pd.read_csv("results.csv")

data = data.sort_values(by=['nnz'])

print(data)

# chart = alt.Chart(data).mark_circle().encode(alt.X('nnz',scale=alt.Scale(type='log'),title="NNZ"), alt.Y('cusparse',scale=alt.Scale(type='log'),title='Runtime (ms)')).properties(title='SPMV Performance')
# chart.configure_title()
# save(chart, 'suitesparse.png')

ax = plt.gca()
data.plot(kind='scatter', x='nnz', y='cusparse',color='red',ax=ax)
data.plot(kind='scatter', x='nnz', y='moderngpu',color='blue',ax=ax)
plt.grid(True)
ax.set_xscale('log')
ax.set_yscale('log')
plt.title("SPMV Performance (SuiteSparse)")
plt.legend(['cuSparse', 'ModernGPU'])
plt.xlabel("NNZ")
plt.ylabel("Runtime (ms)")
plt.savefig('suitesparse.png')