import pandas as pd
import os

datadir = "/home/jwapman/spmv/profiles/"
outfile = "parsed_results.csv"

suitesparse_data = pd.DataFrame()
column_names = ""

for filename_short in os.listdir(datadir):
  filename_long = datadir + filename_short

  # Discard file if the first line contains "Error"
  with open(filename_long) as f:
    first_line = f.readline()
    if "Error" in first_line:
      continue

  # Get the file data and ignore the 
  filedata = pd.read_csv(filename_long, skiprows=5)
  # print(filedata)

  # Building the column headers for the first time
  if suitesparse_data.empty:
    column_names = filedata["Metric Name"].tolist()
    column_names = ["dataset"] + column_names
    suitesparse_data = pd.DataFrame(columns=column_names)

  # Create a new row in the dataframe
  # Not necessarily the most efficient way to do this. Optimize for future speed
  dataset = [os.path.splitext(filename_short)[0]]
  metrics = filedata["Avg"].tolist()
  new_row = dataset + metrics
  suitesparse_data.loc[len(suitesparse_data)] = new_row 
  # suitesparse_data.append(new_row)

print(suitesparse_data)
suitesparse_data.to_csv(outfile)