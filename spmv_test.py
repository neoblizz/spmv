#!/usr/bin/env python3
#SBATCH -p bowser -c 1 --gpus=V100:1

import subprocess
import os
from datetime import datetime

def strip_path(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]

# Setup Paths for binary and datasets
BIN = "./tests/spmv/bin/spmv"
DATASET_BASE = "/data/suitesparse_dataset/MM/"
DATASET = ""

# Search the dataset tree for all .mtx files
if os.path.exists("datasets.txt"):
  os.remove("datasets.txt")

find_dataset_command = "find " + DATASET_BASE + DATASET + " -type f -name \"*.mtx\" > datasets.txt"
print(find_dataset_command)
subprocess.run(find_dataset_command, shell=True)

now = datetime.now()

RESULTS_FILE = "results_" + now.strftime("%Y%m%d_%H:%M:%S") + ".csv"
print(RESULTS_FILE)

results = open(RESULTS_FILE, "w")

results.write("File,rows,cols,nnz,cusparse\n")

PROFILEDIR = "profiles_" + now.strftime("%Y%m%d_%H:%M:%S")
os.mkdir(PROFILEDIR)

with open("datasets.txt", "r") as datasets:
  for dataset in datasets:
    benchmark_cmd = BIN + " " + dataset.rstrip() + " | tail -n 1 > temp_spmvbenchmark.txt"
    print(benchmark_cmd)
    retval = subprocess.run(benchmark_cmd, shell=True)

    if retval.returncode == 0:
      subprocess.run("cat temp_spmvbenchmark.txt >> " + RESULTS_FILE, shell=True)

      # Do profiling
      MTXNAME = strip_path(dataset)
      print("Profiling " + MTXNAME)

      profile_cmd = "nvprof --kernels load_balancing_kernel -m all --csv --log-file " + PROFILEDIR + "/" + MTXNAME + ".log " +  BIN + " " + dataset
      subprocess.run(profile_cmd, shell=True)

os.remove("temp_spmvbenchmark.txt")

print("All Tests Completed")


