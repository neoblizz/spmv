#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include "test_utils.h"

namespace cg = cooperative_groups;

template<typename index_t = int>
__global__ void spmv_tiled_kernel(index_t num_rows) {
  // Each block sets the shared memory region as the output, initialized to 0
  // Iterate up to the tile boundary

  // Use Ampere's CUDAMemCpyAsynch
  // Need to use cuda cooperative groups

  // Start with the very basic tiled iteration first, then add fancy Ampere
  // stuff later
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Num Rows: %d\n", num_rows);
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
      &numBlocksPerSm, spmv_tiled_kernel<int>, numThreads, sharedmem))
  // launch
    // float *input_ptr = thrust::raw_pointer_cast(input.data());
    printf("Rows: %d\n", A.num_rows);
  void *kernelArgs[] = {(void*)&A.num_rows};
  dim3 dimBlock(numThreads, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

  // execute SpMV
  Timer t;
  t.start();
  CHECK_CUDA(cudaLaunchCooperativeKernel((void *)spmv_tiled_kernel<int>, dimGrid,
                                         dimBlock, kernelArgs, sharedmem))

  cudaDeviceSynchronize();
  t.stop();

  return t.elapsed();
}