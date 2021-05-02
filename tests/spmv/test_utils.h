#pragma once

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.

#include <iostream>
#include <iomanip>
#include <string>
#include "time.h"
#include "sys/time.h"

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

template <typename T>
static bool equal(T f1, T f2)
{
    // return(std::fabs(f1 - f2) <= 1e-4);
    // T eps = std::numeric_limits<T>::epsilon();
    T eps = 0.1;
    return (std::fabs(f1 - f2) <=
            eps *
                std::fmax(std::fabs(f1), std::fabs(f2)));
}

template <typename vector_t, typename index_t, typename value_t>
void cpu_spmv(csr_t<index_t, value_t> &A, vector_t &x, vector_t &y)
{
    // Loop over all the rows of A
    for (index_t row = 0; row < A.num_rows; row++)
    {
        y[row] = 0.0;
        // Loop over all the non-zeroes within A's row
        for (auto k = A.row_offsets[row];
             k < A.row_offsets[row + 1]; ++k)
            y[row] += A.nonzero_vals[k] * x[A.col_idx[k]];
    }
}

template <typename vector_t>
bool validate(vector_t &a, vector_t &b)
{
    for (size_t i = 0; i < a.size() - 1; i++)
    {
        if (!equal(a[i], b[i]))
        {
            std::cout << "i = " << i << ": " << std::setprecision(20) << a[i]
                      << " != " << b[i] << std::endl;
            std::cout << "Error = " << std::fabs(a[i] - b[i]) << std::endl;
            double error_percent = std::fabs(a[i] - b[i]) / std::fmax(std::fabs(a[i]), std::fabs(b[i]));
            std::cout << "Error % = " << error_percent << std::endl;
            return false;
        }
    }

    return true;
}

class Timer
{
public:
    Timer()
    {
        time_start = getTime();
        time_stop = getTime();
    }
    void start()
    {
        time_start = getTime();
    }
    void stop()
    {
        time_stop = getTime();
    }
    double elapsed()
    {
        // Return elapsed time in ms
        return time_stop - time_start;
    }

    double getTime()
    {
        struct timeval tv;
        gettimeofday(&tv, 0);
        return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    }

private:
    double time_start, time_stop; // time in ms
    double elapsed_time;
};

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
        if (threadIdx.x != 0)
        {
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