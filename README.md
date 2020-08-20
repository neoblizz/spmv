# Load-Balanced Sparse Matrix-Vector Multiplication (SpMV)
Efficient Sparse Matrix-Vector Multiplication (SpMV) using ModernGPU (MTX + CSR formats). 
For more details, please see [moderngpu](https://github.com/moderngpu/moderngpu/wiki/Load-balancing-search#sparse-matrix--vector) 
for more details. This repository is a simple wrapper around moderngpu's implementation with 
support for loading matrix market format (MTX) using compressed sparse row (CSR) format.
