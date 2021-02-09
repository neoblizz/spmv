#pragma once

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.

#include <iostream>
#include <string>
#include "time.h"
#include "sys/time.h"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
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
void cpu_spmv(csr_t<index_t, value_t> &A, vector_t x, vector_t y)
{
    auto values = A.Ax.data();
    auto indices = A.Aj.data();
    auto offsets = A.Ap.data();

    // Loop over all the rows of A
    for (index_t row = 0; row < A.num_rows; row++)
    {
        y[row] = 0.0;
        // Loop over all the non-zeroes within A's row
        for (auto k = offsets[row];
             k < offsets[row + 1]; ++k)
            y[row] += values[k] * x[indices[k]];
    }
}

template <typename vector_t>
bool validate(vector_t a, vector_t b)
{
    for (size_t i = 0; i < a.size() - 1; i++)
    {
        if (!equal(a[i], b[i]))
        {
            std::cout << "i = " << i << ": " << a[i]
                      << " != " << b[i] << std::endl;
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

    double getTime() {
        struct timeval tv;
        gettimeofday(&tv, 0);
        return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    }

private:
    double time_start, time_stop; // time in ms
    double elapsed_time;
};
