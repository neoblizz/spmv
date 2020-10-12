# Setup Paths for binary and datasets
BIN=./bin/spmm
PATH=/home/mosama/alt-mosama/suitesparse
DATASET=../../datasets
SUITES=$DATASET/suitesparse.txt
# Perform evaluation
echo "dataset,nnz,elapsed" >> results_suite.csv
while IFS= read -r line
do
  echo $BIN $PATH/$line/$line.mtx
  $BIN $PATH/$line/
