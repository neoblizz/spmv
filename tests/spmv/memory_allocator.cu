#include <iostream>
#include <string>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

// IMPORTANT NOTE: This class does not incorporate any synchronization or shared state between threads.
// Accordingly, the programmer must take care to ensure serialization OUTSIDE this class, or alternatively
// ensure that a single thread allocates memory on behalf of the entire block
class MemoryAllocator
{
public:
    __device__ MemoryAllocator(uintptr_t *_base_ptr, size_t _size_bytes) : base_ptr(_base_ptr), cur_ptr(_base_ptr), total_bytes(_size_bytes), allocated_bytes(0)
    {
    }

    __device__ __forceinline__ void init(uintptr_t *_base_ptr, size_t _size_bytes)
    {
        base_ptr = _base_ptr;
        cur_ptr = _base_ptr;
        total_bytes = _size_bytes;
        allocated_bytes = 0;
    }

    __device__ __forceinline__ void reset()
    {
        cur_ptr = base_ptr;
        allocated_bytes = 0;
    }

    // Allocate memory. Note that for parallel programming, any locks must be done _outside_ of this function
    template <typename T>
    __device__ __forceinline__ T *allocate(size_t _size_elem)
    {
        if(threadIdx.x != 0) {
            printf("WARNING ThreadIdx.x != 0\n");
        }
        
        // Check if there is enough memory left. NOTE: need to convert to bytes
        if ((_size_elem * sizeof(T)) > size_remaining_bytes())
        {
            printf("ERROR Block %d has 0 ... bytes remaining\n", blockIdx.x);
            return NULL;
        }
        else
        {
            printf("Processing request of size %ld\n", _size_elem);
            T *ret_ptr = (int *)cur_ptr;

            cur_ptr = (uintptr_t *)&ret_ptr[_size_elem];

            // Increment by the number of bytes we just allocated
            allocated_bytes += (_size_elem * sizeof(T));

            printf("Returning %p, updating to %p\n", ret_ptr, cur_ptr);
            return ret_ptr;
        }
    }

    // Returns the allocated size in bytes
    __device__ __forceinline__ size_t size_allocated_bytes() { return allocated_bytes; }

    template <typename T>
    __device__ __forceinline__ size_t size_allocated_elems() { return size_allocated_bytes() / sizeof(T); }

    // Returns the remaining size in bytes
    __device__ __forceinline__ size_t size_remaining_bytes()
    {
        return total_bytes - allocated_bytes;
    }

    template <typename T>
    __device__ __forceinline__ size_t size_remaining_elems()
    {
        return size_remaining_bytes() / sizeof(T);
    }

    // Return the number of bytes managed by the memory allocator
    __device__ __forceinline__ size_t size_total_bytes() { return total_bytes; }

    template <typename T>
    __device__ __forceinline__ size_t size_total_elems() { return size_total_bytes() / sizeof(T); }

private:
    uintptr_t *base_ptr;
    uintptr_t *cur_ptr;
    size_t total_bytes;
    size_t allocated_bytes;
};

__global__ void test_kernel(size_t shmemPerBlock)
{
    extern __shared__ size_t shmem[];

    MemoryAllocator allocator(shmem, shmemPerBlock);

    printf("Shmem = %p\n", shmem);

    printf("Total (bytes): %ld\n", allocator.size_total_bytes());
    printf("Remaining (bytes): %ld\n", allocator.size_remaining_bytes());

    printf("\n=====\n");

    size_t first_size = (allocator.size_total_bytes() / 2) / sizeof(int);
    printf("First Size (elements) = %ld\n", first_size); // NOTE this works

    int *first = allocator.allocate<int>(first_size);
    printf("First = %p\n", first);
    printf("Remaining (bytes): %ld\n", allocator.size_remaining_bytes());

    for (int i = 0; i < first_size; i++)
    {
        first[i] = i;
    }

    size_t second_size = allocator.size_remaining_bytes() / sizeof(int);
    printf("Second Size (elements) = %ld\n", second_size);

    int *second = allocator.allocate<int>(allocator.size_remaining_bytes() / sizeof(int));
    printf("Second = %p\n", first);
    printf("Remaining (bytes): %ld\n", allocator.size_remaining_bytes());

    for (int i = 0; i < second_size; i++)
    {
        second[i] = i;
    }
}

int main(int argc, char **argv)
{
    /* ========== Setup Device Properties ========== */
    int device = 0;
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

    // Setup grid and block properties
    int numBlocksPerSm = 0;
    int numThreadsPerBlock = 0;
    int shmemPerBlock = 0; // bytes

    // Use the max number of threads per block to maximize parallelism over shmem
    numThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    shmemPerBlock = deviceProp.sharedMemPerBlockOptin;

    printf("Threads Per Block: %d\n", numThreadsPerBlock);
    printf("Shmem Per Block (bytes): %d\n", shmemPerBlock);

    CHECK_CUDA(cudaFuncSetAttribute(test_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    shmemPerBlock));

    // Need to know the max occupancy to determine how many blocks to launch for
    // the cooperative kernel. All blocks must be resident on SMs
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, test_kernel, numThreadsPerBlock,
        shmemPerBlock))

    printf("Blocks per SM: %d\n", numBlocksPerSm);

    //   dim3 dimBlock(numThreadsPerBlock, 1, 1);
    //   dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);
    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(1, 1, 1);

    /* ========== Setup Kernel Call ========== */
    void *kernelArgs[] = {&shmemPerBlock};

    /* ========== SETUP AMPERE CACHE PINNING ========== */
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream)); // Create CUDA stream

    /* ========== Execute ========== */
    CHECK_CUDA(cudaLaunchCooperativeKernel((void *)test_kernel,
                                           dimGrid, dimBlock, kernelArgs,
                                           shmemPerBlock, stream))

    CHECK_CUDA(cudaDeviceSynchronize())

    return 0;
}