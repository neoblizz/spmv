#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include "test_utils.h"

namespace cg = cooperative_groups;

template <typename index_t = int, typename value_t = float>
__global__ void spmv_tiled_kernel(index_t num_rows, index_t num_cols,
                                  index_t num_nonzeros, index_t *row_offsets,
                                  index_t *col_idx, value_t *nonzeros,
                                  value_t *input, value_t *output) {
  // Store the output in shared memory
  extern __shared__ value_t output_shmem[];
  // Each block sets the shared memory region as the output, initialized to 0
  // Iterate up to the tile boundary

  // Use Ampere's CUDAMemCpyAsynch
  // Need to use cuda cooperative groups

  cg::grid_group grid = cg::this_grid();
  grid.sync();
}

template <typename index_t = int, typename value_t = float, typename dinput_t,
          typename doutput_t>
double spmv_tiled(csr_t<index_t, value_t> &A, dinput_t &input,
                  doutput_t &output) {
  /* ========== Setup Device Properties ========== */
  int device = 0;
  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

  // Setup grid and block properties
  int numBlocksPerSm = 0;
  int numThreadsPerBlock = 0;
  int shmemPerBlock = 0; // bytes

  // Use the max number of threads per block to maximize parallelism over shmem
  numThreadsPerBlock = deviceProp.maxThreadsPerBlock/2;

  // Host code
  int carveout = 100; // prefer shared memory capacity 100% of maximum
  // Named Carveout Values:
  // carveout = cudaSharedmemCarveoutDefault;   //  (-1)
  // carveout = cudaSharedmemCarveoutMaxL1;     //   (0)
  // carveout = cudaSharedmemCarveoutMaxShared; // (100)



  // cudaFuncSetAttribute(spmv_tiled_kernel<int,float>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);

  shmemPerBlock = deviceProp.sharedMemPerBlockOptin;

  cudaFuncSetAttribute(spmv_tiled_kernel<int,float>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmemPerBlock);

  int rows_per_block = (shmemPerBlock / sizeof(value_t)) / 2;
  printf("Rows Per Block: %d\n", rows_per_block);
  printf("Shmem Per Block: %d\n", shmemPerBlock);

  // Need to know the max occupancy to determine how many blocks to launch for
  // the cooperative kernel. All blocks must be resident on SMs
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, spmv_tiled_kernel<int, float>, numThreadsPerBlock,
      shmemPerBlock))

  printf("Blocks per SM: %d\n", numBlocksPerSm);

  // CHECK_CUDA(cudaFuncSetAttribute((void *)spmv_tiled_kernel<int,float>,
  // cudaFuncAttributePreferredSharedMemoryCarveout,
  // cudaDevAttrMaxSharedMemoryPerMultiprocessor))
            
  //                      printf("Max shmem: %d\n", cudaDevAttrMaxSharedMemoryPerBlockOptin);

      /* ========== Setup Kernel Call ========== */
      void *row_offsets = thrust::raw_pointer_cast(A.d_Ap.data());
  void *col_idx = thrust::raw_pointer_cast(A.d_Aj.data());
  void *nonzeros = thrust::raw_pointer_cast(A.d_Ax.data());
  void *input_ptr = thrust::raw_pointer_cast(input.data());
  void *output_ptr = thrust::raw_pointer_cast(output.data());
  void *kernelArgs[] = {&A.num_rows,  &A.num_columns, &A.num_nonzeros,
                        &row_offsets, &col_idx,       &nonzeros,
                        &input_ptr,   &output_ptr};
  dim3 dimBlock(numThreadsPerBlock, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

  /* ========== Execute SPMV ========== */
  Timer t;
  t.start();
  CHECK_CUDA(cudaLaunchCooperativeKernel((void *)spmv_tiled_kernel<int, float>,
                                         dimGrid, dimBlock, kernelArgs,
                                         shmemPerBlock, 0))

  CHECK_CUDA(cudaDeviceSynchronize())
  t.stop();

  return t.elapsed();
}