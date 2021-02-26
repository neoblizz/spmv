#pragma once

#include <cub/cub.cuh> // or equivalently <cub/device/device_spmv.cuh>
// Declare, allocate, and initialize device-accessible pointers for input matrix A, input vector x,
// and output vector y

#include "test_utils.h"
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.

template <typename index_t = int, typename value_t = float,
          typename dinput_t, typename doutput_t>
double spmv_cub(csr_t<index_t, value_t> &A, dinput_t &input, doutput_t &output)
{

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, A.d_Ax.data().get(),
                         A.d_Ap.data().get(), A.d_Aj.data().get(), input.data().get(), output.data().get(),
                         A.num_rows, A.num_columns, A.num_nonzeros);
  cudaDeviceSynchronize();
  // Allocate temporary storage
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes))
  // Run SpMV

  Timer t;
  t.start();
  cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, A.d_Ax.data().get(), A.d_Ap.data().get(), A.d_Aj.data().get(), input.data().get(), output.data().get(),
                         A.num_rows, A.num_columns, A.num_nonzeros);
  cudaDeviceSynchronize();
  t.stop();
  CHECK_CUDA(cudaFree(d_temp_storage))

  return t.elapsed();
}