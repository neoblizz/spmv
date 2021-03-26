#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include "test_utils.h"

namespace cg = cooperative_groups;

template <typename index_t = int, typename value_t = float>
__global__ void spmv_tiled_kernel(index_t num_rows, index_t num_cols,
                                  index_t num_nonzeros, value_t *row_offsets) {
  // Each block sets the shared memory region as the output, initialized to 0
  // Iterate up to the tile boundary

  // Use Ampere's CUDAMemCpyAsynch
  // Need to use cuda cooperative groups

  // Start with the very basic tiled iteration first, then add fancy Ampere
  // stuff later
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Num Rows: %d %d %d\n", num_rows, num_cols, num_nonzeros);
  }

  cg::grid_group grid = cg::this_grid();
  grid.sync();
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
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

  // Statistics about the GPU device
  printf(
      "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
      deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  int sharedmem = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, spmv_tiled_kernel<int, float>, numThreads, sharedmem))
  // launch
  void *input_ptr = thrust::raw_pointer_cast(input.data());
  void *kernelArgs[] = {(void *)&A.num_rows, (void *)&A.num_columns,
                        (void *)&A.num_nonzeros, (void*)&input_ptr};
  dim3 dimBlock(numThreads, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

  // execute SpMV
  Timer t;
  t.start();
  CHECK_CUDA(cudaLaunchCooperativeKernel(
      (void *)spmv_tiled_kernel<int, float>, dimGrid, dimBlock, kernelArgs, sharedmem))

  CHECK_CUDA(cudaDeviceSynchronize())
  t.stop();

  return t.elapsed();
}