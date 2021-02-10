#!/bin/sh

#SBATCH -p daisy -c 1 --gpus=V100:1

# Setup Paths for binary and datasets
BIN=./tests/spmv/bin/spmv
DATASET=/data/suitesparse_dataset/MM/DIMACS10

rm datasets.txt
find $DATASET -type f -name "*.mtx" > datasets.txt

rm results.csv
touch results.csv

echo "File,rows,cols,nnz,moderngpu,cusparse,cub" >> results.csv
while IFS= read -r line; do
  echo "$line"
  $BIN $line | tail -n 1 > temp.txt

  # Append to results.csv on successful exit
  exit_status=$?
  if [ $exit_status -eq 0 ]; then
      cat temp.txt >> results.csv
  fi

done < datasets.txt
rm temp.txt

echo "All Tests Completed"

python3 ./plot.py
