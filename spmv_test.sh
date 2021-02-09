#!/bin/sh

#SBATCH -p daisy -c 1 --gpus=V100:1

# Setup Paths for binary and datasets
BIN=./tests/spmv/bin/spmv
DATASET=/data/suitesparse_dataset/MM/

rm datasets.txt
find $DATASET -type f -name "*.mtx" > datasets.txt

rm results.csv
touch results.csv

while IFS= read -r line; do
  echo "$line"
  $BIN $line | tail -n 1 >> results.csv
done < datasets.txt

# Perform evaluation
# echo "dataset,nnz,elapsed" >> results_suite.csv
# while IFS= read -r line
# do
#   echo $BIN $PATH/$line/$line.mtx
#   $BIN $PATH/$line/
echo "Done"