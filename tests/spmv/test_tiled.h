#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include "test_utils.h"

namespace cg = cooperative_groups;

template <typename index_t>
__device__ __forceinline__ index_t global2tile(const index_t &global_idx,
                                               const index_t &tile_size)
{
  // Note that this function assumes a valid global index and does not attempt
  // to perform bounds checking for partial tiles

  // Tile_index + offset_within_tile
  index_t local_idx = (global_idx / tile_size) + (global_idx % tile_size);

  return local_idx;
}

template <typename index_t>
__device__ __forceinline__ index_t tile2global(const index_t &local_idx,
                                               const index_t &tile_idx,
                                               const index_t &tile_size)
{
  // Note that this function does not attempt to perform bounds checking for the
  // final tile

  index_t global_idx = local_idx + (tile_idx * tile_size);

  return global_idx;
}

template <typename index_t = int, typename value_t = float>
class TileIterator
{
public:
  __device__ TileIterator(const index_t _num_rows, const index_t _num_cols,
                          const index_t _num_nonzeros,
                          const index_t *_row_offsets, const index_t *_col_idx,
                          const value_t *_nonzeros, const value_t *_input,
                          value_t *_output, const index_t _rows_per_block_tile,
                          const index_t _tile_col_size,
                          index_t *_local_row_offsets, index_t *_lb_stats,
                          const bool _store_end_offsets_in_shmem,
                          const bool _debug)
      : num_rows(_num_rows),
        num_cols(_num_cols),
        num_nonzeros(_num_nonzeros),
        row_offsets(_row_offsets),
        col_idx(_col_idx),
        nonzeros(_nonzeros),
        input(_input),
        output(_output),
        tile_col_size(_tile_col_size),
        store_end_offsets_in_shmem(_store_end_offsets_in_shmem),
        debug(_debug)
  {
    MemoryAllocator allocator((size_t *)_local_row_offsets, (size_t)(_rows_per_block_tile *2* sizeof(index_t)));

    rows_per_block_tile = _rows_per_block_tile;
    rows_per_gpu_tile = _rows_per_block_tile * gridDim.x;

    cur_row_tile_idx = 0;
    cur_col_tile_idx = 0;

    if (threadIdx.x == 0)
    {
      local_row_offsets_start = (int *)allocator.allocate(size_t(rows_per_block_tile * sizeof(index_t)));
      local_row_offsets_end = (int *)allocator.allocate(size_t(rows_per_block_tile * sizeof(index_t)));
    }
    lb_stats = _lb_stats;
  }

  __device__ __forceinline__ index_t
  row_elems_in_tile(const index_t &global_row_idx, const index_t &block_row_idx,
                    const index_t &tile_boundary)
  {
    assert(store_end_offsets_in_shmem);

    index_t total_remaining_nonzeros = 0;

    index_t begin = local_row_offsets_start[block_row_idx];
    index_t end = local_row_offsets_end[block_row_idx];

    while (begin < end)
    {
      index_t mid = floor((begin + end) / 2);

      index_t col_key = col_idx[mid];

      if (col_key > tile_boundary)
      {
        end = mid;
      }
      else
      {
        begin = mid + 1;
      }
    }

    index_t actual_end = end - 1;

    return (actual_end - local_row_offsets_start[block_row_idx]);
  }

  __device__ __forceinline__ bool all_tiles_finished()
  {
    // How many evenly-sized tiles are there?
    index_t number_of_gpu_tiles_in_matrix = (num_rows / rows_per_gpu_tile);

    // Remainder tile
    if (num_rows % rows_per_gpu_tile)
      number_of_gpu_tiles_in_matrix++;

    // cur_row_tile_idx incremented only after the primary tile is finished
    // (with all its member column tiles)
    if (cur_row_tile_idx >= number_of_gpu_tiles_in_matrix)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  __device__ __forceinline__ void load_primary_tile()
  {
    if (blockIdx.x == 0 && threadIdx.x == 0 && debug)
    {
      printf("Loading Metadata for tile (%d,...) into shmem\n",
             cur_row_tile_idx);
    }
    // Need to simultaneously keep track of the current row in the tile as well
    // as the row index in the global coordinates

    int cur_row_in_gpu_tile = blockIdx.x * rows_per_block_tile + threadIdx.x;
    int cur_row_in_matrix =
        tile2global(cur_row_in_gpu_tile, cur_row_tile_idx, rows_per_gpu_tile);

    int cur_row_in_block_tile = threadIdx.x;

    int stride = blockDim.x;

    // Iterate over all rows in the current tile
    for (; cur_row_in_matrix < num_rows &&
           cur_row_in_block_tile < rows_per_block_tile;
         cur_row_in_matrix += stride, cur_row_in_block_tile += stride,
         cur_row_in_gpu_tile += stride)
    {
      local_row_offsets_start[cur_row_in_block_tile] =
          row_offsets[cur_row_in_matrix];

      if (store_end_offsets_in_shmem)
      {
        local_row_offsets_end[cur_row_in_block_tile] =
            row_offsets[cur_row_in_matrix + 1];
      }
      // printf(
      //     "Block %d Loading matrix row %d block tile idx %d gpu tile idx %d "
      //     "offset %d\n",
      //     blockIdx.x, cur_row_in_matrix, cur_row_in_block_tile,
      //     cur_row_in_gpu_tile, local_row_offsets[cur_row_in_block_tile]);
    }

    __syncthreads();
  }

  __device__ __forceinline__ void load_secondary_tile(){};

  __device__ __forceinline__ void evict_primary_tile() {}

  __device__ __forceinline__ void evict_secondary_tile()
  {
    // In the src-first implementation, there is nothing to do for this function
    // except maybe resetting the L2 cache
  }

  __device__ __forceinline__ bool primary_tile_finished()
  {
    // How many evenly-sized tiles are there?
    index_t number_of_tiles_in_matrix = (num_cols / tile_col_size);

    // Remainder tile
    if (num_cols % tile_col_size)
      number_of_tiles_in_matrix++;

    if (cur_col_tile_idx >= number_of_tiles_in_matrix)
    {
      return true;
    }
    else
    {
      return false;
    }
  };

  __device__ __forceinline__ void process_all_tiles()
  {
    while (!all_tiles_finished())
    {
      load_primary_tile();
      process_primary_tile();
      // evict_primary_tile();
    }
  }

  __device__ __forceinline__ void process_primary_tile()
  {
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //   printf("Processing Tile (%d,...)\n", cur_row_tile_idx);
    // }
    while (!primary_tile_finished())
    {
      // load_secondary_tile();
      process_secondary_tile();
      // evict_secondary_tile();
    }

    cur_row_tile_idx++;
    cur_col_tile_idx = 0;
  }

  __device__ __forceinline__ void lb_warp_per_row() {}

  __device__ __forceinline__ void lb_thread_per_row()
  {
    cg::grid_group grid = cg::this_grid();

    if (debug)
    {
      lb_stats[blockIdx.x] = 0;
      __syncthreads();
    }

    int cur_row_in_gpu_tile = blockIdx.x * rows_per_block_tile + threadIdx.x;
    int cur_row_in_matrix =
        tile2global(cur_row_in_gpu_tile, cur_row_tile_idx, rows_per_gpu_tile);

    int cur_row_in_block_tile = threadIdx.x;

    int stride = blockDim.x;

    // End of the col tile boundary
    index_t tile_boundary =
        min(num_cols, (cur_col_tile_idx + 1) * tile_col_size);

    // Iterate over all rows in the current tile
    for (; cur_row_in_matrix < num_rows &&
           cur_row_in_block_tile < rows_per_block_tile;
         cur_row_in_matrix += stride, cur_row_in_block_tile += stride,
         cur_row_in_gpu_tile += stride)
    {
      // Process a row

      printf(
          "Block %d, Thread %d, Global Row %d, Local Row %d, Tile Nonzeros "
          "%d\n",
          blockIdx.x, threadIdx.x, cur_row_in_matrix, cur_row_in_block_tile,
          row_elems_in_tile(cur_row_in_matrix, cur_row_in_block_tile,
                            tile_boundary));

      value_t sum = 0.0;
      index_t offset = local_row_offsets_start[cur_row_in_block_tile];

      index_t max_offset;

      if (store_end_offsets_in_shmem)
      {
        max_offset = local_row_offsets_end[cur_row_in_block_tile];
      }
      else
      {
        max_offset = row_offsets[cur_row_in_block_tile + 1];
      }

      while (true)
      {
        if (offset >= max_offset)
          break;

        index_t col = col_idx[offset];

        if (col >= tile_boundary)
        {
          // printf("Col %d greater than boundary %d\n", col, tile_boundary);
          break;
        }
        else
        {
          // printf("Processing col %d\n", col);
        }
        // atomicAdd(&block_nonzeros, 1);
        sum += nonzeros[offset] * input[col];

        if (debug)
          atomicAdd(&lb_stats[blockIdx.x], 1);

        offset++;
      }

      // Finished with the row

      // Save the offset for the next iteration
      local_row_offsets_start[cur_row_in_block_tile] = offset;
      if (sum != 0)
      {
        output[cur_row_in_matrix] += sum;
      }
    }

    // Must sync at the end of the tile to preserve cache reuse

    grid.sync();

    if (threadIdx.x == 0 && debug)
    {
      printf("Tile (%d,%d) block %d has %d nonzeros\n", cur_row_tile_idx,
             cur_col_tile_idx, blockIdx.x, lb_stats[blockIdx.x]);
    }

    cur_col_tile_idx++;
  }

  __device__ __forceinline__ void process_secondary_tile()
  {
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //   printf("Processing Tile (%d,%d)\n", cur_row_tile_idx,
    //   cur_col_tile_idx);
    // }

    lb_thread_per_row();
    // lb_warp_per_row();
  }

private:
  // SPMV operator properties
  const index_t num_rows;
  const index_t num_cols;
  const index_t num_nonzeros;
  const index_t *row_offsets;
  const index_t *col_idx;
  const value_t *nonzeros;
  const value_t *input;
  value_t *output;

  // Tiling metadata
  index_t rows_per_block_tile;
  index_t rows_per_gpu_tile;
  index_t tile_col_size;

  index_t cur_row_tile_idx;
  index_t cur_col_tile_idx;

  // shmem
  index_t *local_row_offsets_start;
  index_t *local_row_offsets_end;

  index_t *lb_stats;

  const bool store_end_offsets_in_shmem;

  const bool debug;
};

template <typename index_t = int, typename value_t = float>
__global__ void spmv_tiled_kernel(
    const index_t num_rows, const index_t num_cols, const index_t num_nonzeros,
    const index_t *row_offsets, const index_t *col_idx, const value_t *nonzeros,
    const value_t *input, value_t *output, const index_t rows_per_block_tile,
    const index_t tile_col_size, index_t *lb_stats,
    const bool store_end_offsets_in_shmem, const bool debug)
{
  // Store the output in shared memory
  extern __shared__ index_t shmem[];

  TileIterator<int, float> iterator(
      num_rows, num_cols, num_nonzeros, row_offsets, col_idx, nonzeros, input,
      output, rows_per_block_tile, tile_col_size, shmem, lb_stats,
      store_end_offsets_in_shmem, debug);

  iterator.process_all_tiles();

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
}

template <typename index_t = int, typename value_t = float, typename dinput_t,
          typename doutput_t>
double spmv_tiled(csr_t<index_t, value_t> &A, dinput_t &input,
                  doutput_t &output, bool debug)
{
  /* ========== Setup Device Properties ========== */
  int device = 0;
  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

  // Setup grid and block properties
  int numBlocksPerSm = 0;
  int numThreadsPerBlock = 0;
  int shmemPerBlock = 0; // bytes

  int target_occupancy = 1;

  // Number of coordinates. TODO calculate this
  // based on architecture L2 properties
  index_t tile_size = 0;

  // Use the max number of threads per block to maximize parallelism over shmem

  numThreadsPerBlock = deviceProp.maxThreadsPerBlock / target_occupancy;
  shmemPerBlock = (deviceProp.sharedMemPerBlockOptin / target_occupancy);

  bool store_end_offsets_in_shmem = true;

  index_t data_elems_per_row = 1;
  if (store_end_offsets_in_shmem)
  {
    data_elems_per_row = 2;
  }
  index_t rows_per_block =
      (shmemPerBlock / (sizeof(index_t) * data_elems_per_row));

  printf("Threads Per Block: %d\n", numThreadsPerBlock);
  printf("Rows Per Block: %d\n", rows_per_block);
  printf("Shmem Per Block (bytes): %d\n", shmemPerBlock);

  CHECK_CUDA(cudaFuncSetAttribute(spmv_tiled_kernel<index_t, value_t>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  shmemPerBlock));

  // Need to know the max occupancy to determine how many blocks to launch for
  // the cooperative kernel. All blocks must be resident on SMs
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, spmv_tiled_kernel<index_t, value_t>, numThreadsPerBlock,
      shmemPerBlock))

  printf("Blocks per SM: %d\n", numBlocksPerSm);

  dim3 dimBlock(numThreadsPerBlock, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

  /* ========== SPACE FOR LOAD BALANCE STATS ========== */
  // Allocate a location for each thread?
  thrust::host_vector<int> h_lb_stats(dimGrid.x, 0);
  thrust::device_vector<int> d_lb_stats = h_lb_stats;

  /* ========== Setup Kernel Call ========== */
  void *row_offsets = thrust::raw_pointer_cast(A.d_row_offsets.data());
  void *col_idx = thrust::raw_pointer_cast(A.d_col_idx.data());
  void *nonzeros = thrust::raw_pointer_cast(A.d_nonzero_vals.data());
  void *input_ptr = thrust::raw_pointer_cast(input.data());
  void *output_ptr = thrust::raw_pointer_cast(output.data());
  void *d_lb_stats_ptr = thrust::raw_pointer_cast(d_lb_stats.data());
  void *kernelArgs[] = {
      &A.num_rows, &A.num_columns, &A.num_nonzeros,
      &row_offsets, &col_idx, &nonzeros,
      &input_ptr, &output_ptr, &rows_per_block,
      &tile_size, &d_lb_stats_ptr, &store_end_offsets_in_shmem,
      &debug};

  /* ========== SETUP AMPERE CACHE PINNING ========== */
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream)); // Create CUDA stream

  // Stream level attributes data structure
  cudaStreamAttrValue stream_attribute;

  if (deviceProp.major > 8)
  {
    // Using Ampere

    size_t size =
        min(int(deviceProp.l2CacheSize), deviceProp.persistingL2CacheMaxSize);

    // size is in bytes. Need to convert to elements
    tile_size = size / sizeof(value_t);

    // set-aside the full L2 cache for persisting accesses or the max allowed
    CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size));

    int num_bytes = (int)input.size();
    size_t window_size = min(deviceProp.accessPolicyMaxWindowSize,
                             num_bytes); // Select minimum of user defined
                                         // num_bytes and max window size.

    // Global Memory data pointer
    stream_attribute.accessPolicyWindow.base_ptr =
        reinterpret_cast<void *>(input_ptr);

    // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.num_bytes =
        input.size() / sizeof(value_t);

    // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitRatio = 0.6;

    // Persistence Property
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;

    // Type of access property on cache miss
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    // Set the attributes to a CUDA Stream
    CHECK_CUDA(cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));
  }
  else
  {
    // Using Volta or below
    printf(
        "WARNING: L2 Cache Management available only for compute capabilities "
        "> 8\n");

    tile_size = (deviceProp.l2CacheSize / 2) / sizeof(value_t);
  }

  printf("Tile Size (elements): %d * %d, %d\n", rows_per_block, dimGrid.x,
         tile_size);

  /* ========== Execute SPMV ========== */
  Timer t;
  t.start();
  CHECK_CUDA(cudaLaunchCooperativeKernel((void *)spmv_tiled_kernel<int, float>,
                                         dimGrid, dimBlock, kernelArgs,
                                         shmemPerBlock, stream))

  CHECK_CUDA(cudaDeviceSynchronize())
  t.stop();

  /* ========== RESET THE GPU ========== */

  if (deviceProp.major > 8)
  {
    // Setting the window size to 0 disable it
    stream_attribute.accessPolicyWindow.num_bytes = 0;

    // Overwrite the access policy attribute to a CUDA Stream
    CHECK_CUDA(cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

    // Remove any persistent lines in L2
    CHECK_CUDA(cudaCtxResetPersistingL2Cache());
  }

  return t.elapsed();
}