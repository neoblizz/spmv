import pandas as pd
import os
import re

datadir = "/home/jwapman/spmv/profiles/"
outfile = "parsed_results.csv"


def strip_path(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]

def make_numeric(val):
  fp_val = re.findall("\d+\.\d+", str(val))
  int_val = re.findall("\d+", str(val))
  if fp_val == [] or int_val == []:
    return val
  else:
    if len(fp_val) > len(int_val):
      return fp_val[0]
    else:
      return int_val[0]


suitesparse_data = pd.DataFrame()
column_names = ""

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
        column_names = ["dataset", "rows", "cols",
                        "nnz", "cuSparse Runtime"] + column_names
        suitesparse_data = pd.DataFrame(columns=column_names)

    # Create a new row in the dataframe
    # Not necessarily the most efficient way to do this. Optimize for future speed
    dataset = os.path.splitext(filename_short)[0]

    # Index the dataframe containing output of the spmv runs for each dataset.
    # Discard the first column which contains the dataset name. We already have this
    mtx_stats = mtxdata.loc[mtxdata["File"] == dataset].values.tolist()[0][1:]

    # Extract the average stats only. Min and Max don't matter since this is only
    # one run. Then combine into a single row
    metrics = filedata["Avg"].tolist()
    new_row = [dataset] + mtx_stats + metrics
    suitesparse_data.loc[len(suitesparse_data)] = new_row

# Strip all non-numerical characters (Such as %, GB/sec, etc...) from all columns except the first

suitesparse_data[suitesparse_data.columns[1:]] = suitesparse_data[suitesparse_data.columns[1:]].applymap(make_numeric)
print(suitesparse_data)
suitesparse_data.to_csv(outfile)
