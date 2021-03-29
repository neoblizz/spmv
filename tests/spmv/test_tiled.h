#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include "test_utils.h"

namespace cg = cooperative_groups;

template <typename index_t = int, typename value_t = float>
class TileIterator {
  public:
  __device__ TileIterator(index_t _num_rows, index_t _num_cols,
                          index_t _num_nonzeros, index_t *_row_offsets,
                          index_t *_col_idx, value_t *_nonzeros, value_t *_input,
                          value_t *_output, index_t _tile_row_size,
                          index_t _tile_col_size) {}

  __device__ bool all_tiles_finished() {
  }

  __device__ void load_primary_tile() {
    // Depending on the implementation, maybe allocate space for the output, or
    // the metadata, or both
  }

  __device__ void load_secondary_tile();

  __device__ void evict_primary_tile() {
    // In the src-first context, this means writing the output back to global memory
  }

  __device__ void evict_secondary_tile() {
    // In the src-first implementation, there is nothing to do for this function
    // except maybe resetting the L2 cach
  }

  __device__ bool primary_tile_finished();

  __device__ void process_all_tiles() {
    while(!all_tiles_finished()) {
      load_primary_tile();
      process_primary_tile();
      evict_primary_tile();
    }
  }

  __device__ void process_primary_tile() {
    while(!primary_tile_finished()) {
      load_secondary_tile();
      process_secondary_tile();
      evict_secondary_tile();
    }
  }

  __device__ void process_secondary_tile() {
    cg::grid_group grid = cg::this_grid();
    grid.sync();
  }


  private:
    index_t num_rows;
    index_t num_cols;
    index_t num_nonzeros;
    index_t *row_offsets;
    index_t *col_idx;
    value_t *nonzeros;
    vaue_t *input;
    value_t *output;
    index_t tile_row_size;
    index_t tile_col_size;
};

template <typename index_t = int, typename value_t = float, typename cudaProp_t>
__global__ void spmv_tiled_kernel(index_t num_rows, index_t num_cols,
                                  index_t num_nonzeros, index_t *row_offsets,
                                  index_t *col_idx, value_t *nonzeros,
                                  value_t *input, value_t *output,
                                  index_t tile_row_size, index_t tile_col_size,
                                  cudaProp_t deviceProp) {
  // Store the output in shared memory
  extern __shared__ value_t shmem[];
  // Each block sets the shared memory region as the output, initialized to 0
  // Iterate up to the tile boundary

  // Use Ampere's CUDAMemCpyAsynch
  // Need to use cuda cooperative groups

  // Simple, single-threaded implementation
  // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   for (int i = 0; i < num_rows; i++) {
  //     value_t y = 0;
  //     for (int k = row_offsets[i]; k < row_offsets[i + 1]; k++) {
  //       y = y + (nonzeros[k] * input[col_idx[k]]);
  //     }
  //     output[i] = y;
  //   }
  // }

  // Grid iterates over rows
  // the initial row that this thread will work on

  // Aamer's approach:
  // 1. Load tile src values into shmem
  // 2.

  // Within a src tile, blocks iterate up to the tile boundary, and then perform
  // a grid sync
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
  int shmemPerBlock = 0;  // bytes

  int target_occupancy = 1;

  size_t tile_size = 100;  // Coordinates

  // Use the max number of threads per block to maximize parallelism over shmem
  numThreadsPerBlock = deviceProp.maxThreadsPerBlock / target_occupancy;
  shmemPerBlock = deviceProp.sharedMemPerBlockOptin / target_occupancy;

  cudaFuncSetAttribute(spmv_tiled_kernel<int, float, cudaDeviceProp>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shmemPerBlock);

  int rows_per_block = (shmemPerBlock / sizeof(value_t)) / 3;
  printf("Threads Per Block: %d\n", numThreadsPerBlock);
  printf("Rows Per Block: %d\n", rows_per_block);
  printf("Shmem Per Block (bytes): %d\n", shmemPerBlock);

  // Need to know the max occupancy to determine how many blocks to launch for
  // the cooperative kernel. All blocks must be resident on SMs
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, spmv_tiled_kernel<int, float, cudaDeviceProp>,
      numThreadsPerBlock, shmemPerBlock))

  printf("Blocks per SM: %d\n", numBlocksPerSm);

  /* ========== Setup Kernel Call ========== */
  void *row_offsets = thrust::raw_pointer_cast(A.d_Ap.data());
  void *col_idx = thrust::raw_pointer_cast(A.d_Aj.data());
  void *nonzeros = thrust::raw_pointer_cast(A.d_Ax.data());
  void *input_ptr = thrust::raw_pointer_cast(input.data());
  void *output_ptr = thrust::raw_pointer_cast(output.data());
  void *kernelArgs[] = {&A.num_rows,  &A.num_columns, &A.num_nonzeros,
                        &row_offsets, &col_idx,       &nonzeros,
                        &input_ptr,   &output_ptr,    &tile_size,
                        &tile_size,   &deviceProp};
  dim3 dimBlock(numThreadsPerBlock, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

  /* ========== Execute SPMV ========== */
  Timer t;
  t.start();
  CHECK_CUDA(cudaLaunchCooperativeKernel(
      (void *)spmv_tiled_kernel<int, float, cudaDeviceProp>, dimGrid, dimBlock,
      kernelArgs, shmemPerBlock, 0))

  CHECK_CUDA(cudaDeviceSynchronize())
  t.stop();

  return t.elapsed();
}