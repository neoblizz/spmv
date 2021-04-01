#pragma once

#include <moderngpu/kernel_segreduce.hxx>
#include "test_utils.h"

// using namespace mgpu;
template <typename launch_arg_t = mgpu::empty_t,
          typename index_t = int, typename value_t = float,
          typename input_t, typename output_t>
double spmv_mgpu(csr_t<index_t, value_t> &A, input_t &input, output_t &output)
{

    // ... GPU SPMV
    // GPU device context, print
    mgpu::standard_context_t context(false);

    auto values = A.d_nonzero_vals.data().get();
    auto indices = A.d_col_idx.data().get();
    auto offsets = A.d_row_offsets.data().get();

    int offsets_size = A.num_rows;
    int nnz = A.num_nonzeros;

    Timer t;
    t.start();
    mgpu::spmv(values, indices, input.data().get(), nnz, offsets, offsets_size, output.data().get(), context);

    // Synchronize the device
    context.synchronize();

    t.stop();

    return t.elapsed();
}