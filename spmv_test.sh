# Setup Paths for binary and datasets
BIN=./tests/spmv/bin/spmv
DATASET=/data/suitesparse_dataset/MM
SUITES=$DATASET/suitesparse_mtx.txt


find $DATASET -type f -name "*.mtx" > datasets.txt

while IFS= read -r line; do
  echo "$line"
  $BIN $line
done < datasets.txt

# Perform evaluation
# echo "dataset,nnz,elapsed" >> results_suite.csv
# while IFS= read -r line
# do
#   echo $BIN $PATH/$line/$line.mtx
#   $BIN $PATH/$line/
