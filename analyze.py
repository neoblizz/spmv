import pandas as pd
import os

datadir = "/home/jwapman/spmv/profiles/"
outfile = "parsed_results.csv"

suitesparse_data = pd.DataFrame()
column_names = ""

def strip_path(filepath):
  base=os.path.basename(filepath)
  return os.path.splitext(base)[0]

# Read in the matrix data
mtxdata = pd.read_csv("results.csv")
mtxdata["File"] = mtxdata["File"].apply(strip_path)
print(mtxdata)

for filename_short in os.listdir(datadir):
  filename_long = datadir + filename_short

  # Discard file if the first line contains "Error"
  with open(filename_long) as f:
    first_line = f.readline()
    if "Error" in first_line:
      continue

  # Get the file data and ignore the 
  filedata = pd.read_csv(filename_long, skiprows=5)

  # Building the column headers for the first time
  if suitesparse_data.empty:
    column_names = filedata["Metric Name"].tolist()
    column_names = ["dataset", "rows", "cols", "nnz", "cuSparse Runtime"] + column_names
    suitesparse_data = pd.DataFrame(columns=column_names)

  # Create a new row in the dataframe
  # Not necessarily the most efficient way to do this. Optimize for future speed
  dataset = os.path.splitext(filename_short)[0]

  mtx_stats = mtxdata.loc[mtxdata["File"]==dataset].values.tolist()[0][1:]

  metrics = filedata["Avg"].tolist()
  new_row = [dataset] + mtx_stats + metrics
  suitesparse_data.loc[len(suitesparse_data)] = new_row 

print(suitesparse_data)
suitesparse_data.to_csv(outfile)