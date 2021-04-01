#pragma once

#include <cusparse.h>         // cusparseSpMV
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <thrust/device_vector.h>

#include "test_utils.h"

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

// Helper code from CUDALibrarySamples
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE

template <typename index_t = int, typename value_t = float,
          typename dinput_t, typename doutput_t>
double spmv_cusparse(csr_t<index_t, value_t> &A, dinput_t &input, doutput_t &output)
{

    // Host problem definition
    value_t alpha = 1.0f;
    value_t beta = 0.0f;

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.num_rows, A.num_columns, A.num_nonzeros,
                                     A.d_row_offsets.data().get(), A.d_col_idx.data().get(), A.d_nonzero_vals.data().get(),
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // Create dense vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A.num_columns, input.data().get(), CUDA_R_32F))
    // Create dense vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A.num_rows, output.data().get(), CUDA_R_32F))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

    // execute SpMV
    Timer t;
    t.start();
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                CUSPARSE_MV_ALG_DEFAULT, dBuffer))

    cudaDeviceSynchronize();
    t.stop();

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))

    return t.elapsed();
}