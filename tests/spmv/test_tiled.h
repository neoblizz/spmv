#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include "test_utils.h"

template <typename index_t=int, typename value_t=float, typename input_t=float,
          typename output_t=float>
__global__ void spmv_tiled_kernel(index_t num_rows, index_t num_columns,
                                  index_t num_nonzeros, index_t *row_offsets,
                                  index_t *col_indices, value_t *nonzeros,
                                  input_t *input, output_t *output) {
  // Each block sets the shared memory region as the output, initialized to 0
  // Iterate up to the tile boundary

  // Use Ampere's CUDAMemCpyAsynch
  // Need to use cuda cooperative groups

  // Start with the very basic tiled iteration first, then add fancy Ampere
  // stuff later
}

template <typename index_t = int, typename value_t = float, typename dinput_t,
          typename doutput_t>
double spmv_tiled(csr_t<index_t, value_t> &A, dinput_t &input,
                  doutput_t &output) {
  int device = 0;

  /// This will launch a grid that can maximally fill the GPU, on the default
  /// stream with kernel arguments
  int numBlocksPerSm = 0;
  // Number of threads my_kernel will be launched with
  int numThreads = 128;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, spmv_tiled_kernel<int,float,float,float>,
                                                numThreads, 0);
  // launch
  void *kernelArgs[] = {/* add kernel args */};
  dim3 dimBlock(numThreads, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);
  int sharedmem = 0;
  cudaLaunchCooperativeKernel((void *)spmv_tiled_kernel<int,float,float,float>, dimGrid, dimBlock, kernelArgs, sharedmem);

  // execute SpMV
  Timer t;
  t.start();
  spmv_tiled_kernel<<<1, 1>>>(A.num_rows, A.num_columns, A.num_nonzeros,
                              A.d_Ap.data().get(), A.d_Aj.data().get(),
                              A.d_Ax.data().get(), input.data().get(),
                              output.data().get());

  cudaDeviceSynchronize();
  t.stop();

  return t.elapsed();
}