#!/bin/bash
# file: run_test.sh
# Specify partition
#SBATCH -p dgxa100_40g_1tb
# Request a single GPU
#SBATCH -G 1
# Request 1 CPU task
#SBATCH -n 1
# Set a time limit (10 sec for higher priority when scheduling)
#SBATCH -t 0:10

srun ~/spmv/tests/spmv/bin/spmv ~/DIMACS10/road_usa/road_usa.mtx
