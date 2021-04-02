#include "thrust/device_vector.h"
#include <iostream>
#include "cub/cub.cuh"  // or equivalently <cub/device/device_spmv.cuh>

#define CHECK_CUDA(func)                                                   \
  {                                                                        \
    cudaError_t status = (func);                                           \
    if (status != cudaSuccess) {                                           \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
             cudaGetErrorString(status), status);                          \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  }

int main(int argc, char** argv) {
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  int num_rows = 1000;
  int num_cols = 1000;
  int num_nonzeros = 20;

  thrust::device_vector<float> d_values(num_nonzeros, 0);
  thrust::sequence(d_values.begin(), d_values.end()); // 0,1,2,...

  // float* d_values;  // e.g., [1, 1, 1, 1, 1, 1, 1, 1,
  //        1, 1, 1, 1, 1, 1, 1, 1,
  //        1, 1, 1, 1, 1, 1, 1, 1]

  thrust::device_vector<int> d_column_indices(num_nonzeros, 0);
  thrust::sequence(d_column_indices.begin(), d_column_indices.end());
  // int*   d_column_indices; // e.g., [1, 3, 0, 2, 4, 1, 5, 0,
  //        4, 6, 1, 3, 5, 7, 2, 4,
  //        8, 3, 7, 4, 6, 8, 5, 7]

  thrust::device_vector<int> d_row_offsets(num_rows, 0);
  thrust::sequence(d_row_offsets.begin(), d_row_offsets.end()); // 0,1,2,...
  // int*   d_row_offsets;    // e.g., [0, 2, 5, 7, 10, 14, 17, 19, 22, 24]

  thrust::device_vector<float> d_vector_x(num_rows, 0);
  thrust::fill(d_vector_x.begin(), d_vector_x.end(), 1);
  // float* d_vector_x;       // e.g., [1, 1, 1, 1, 1, 1, 1, 1, 1]

  thrust::device_vector<float> d_vector_y(num_rows, 0);
  // float* d_vector_y;       // e.g., [ ,  ,  ,  ,  ,  ,  ,  ,  ]

  CHECK_CUDA(cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes,
                                    d_values.data().get(), d_row_offsets.data().get(), d_column_indices.data().get(),
                                    d_vector_x.data().get(), d_vector_y.data().get(), num_rows, num_cols,
                                    num_nonzeros, 0, true));

                                    CHECK_CUDA(cudaDeviceSynchronize());

          
  // Allocate temporary storage
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  printf("Allocated %d bytes of temp storage\n", temp_storage_bytes);

  // Run SpMV
  CHECK_CUDA(cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes,
                                    d_values.data().get(), d_row_offsets.data().get(), d_column_indices.data().get(),
                                    d_vector_x.data().get(), d_vector_y.data().get(), num_rows, num_cols,
                                    num_nonzeros, 0, true));

  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaFree(d_temp_storage));
}
