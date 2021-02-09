#pragma once

#include <cub/cub.cuh>   // or equivalently <cub/device/device_spmv.cuh>
// Declare, allocate, and initialize device-accessible pointers for input matrix A, input vector x,
// and output vector y
int    num_rows = 9;
int    num_cols = 9;
int    num_nonzeros = 24;
float* d_values;  // e.g., [1, 1, 1, 1, 1, 1, 1, 1,
                  //        1, 1, 1, 1, 1, 1, 1, 1,
                  //        1, 1, 1, 1, 1, 1, 1, 1]
int*   d_column_indices; // e.g., [1, 3, 0, 2, 4, 1, 5, 0,
                         //        4, 6, 1, 3, 5, 7, 2, 4,
                         //        8, 3, 7, 4, 6, 8, 5, 7]
int*   d_row_offsets;    // e.g., [0, 2, 5, 7, 10, 14, 17, 19, 22, 24]
float* d_vector_x;       // e.g., [1, 1, 1, 1, 1, 1, 1, 1, 1]
float* d_vector_y;       // e.g., [ ,  ,  ,  ,  ,  ,  ,  ,  ]
...
// Determine temporary device storage requirements
void*    d_temp_storage = NULL;
size_t   temp_storage_bytes = 0;
cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
    d_row_offsets, d_column_indices, d_vector_x, d_vector_y,
    num_rows, num_cols, num_nonzeros, alpha, beta);
// Allocate temporary storage
cudaMalloc(&d_temp_storage, temp_storage_bytes);
// Run SpMV
cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
    d_row_offsets, d_column_indices, d_vector_x, d_vector_y,
    num_rows, num_cols, num_nonzeros, alpha, beta);
// d_vector_y <-- [2, 3, 2, 3, 4, 3, 2, 3, 2]