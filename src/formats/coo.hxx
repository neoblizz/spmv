#pragma once

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/**
 * @brief Coordinate (COO) format.
 * 
 * @tparam index_t 
 * @tparam value_t
 */
template<typename index_t = int, 
         typename value_t = float>
struct coo_t {
    index_t num_rows;
    index_t num_columns;
    index_t num_nonzeros;

    thrust::host_vector<index_t> I; // row indices
    thrust::host_vector<index_t> J; // column indices
    thrust::host_vector<value_t> V; // nonzero values

    coo_t() : 
        num_rows(0),
        num_columns(0),
        num_nonzeros(0) { }

    coo_t(index_t r, index_t c, index_t nnz) :
        num_rows(r),
        num_columns(c),
        num_nonzeros(nnz),
        I(nnz),
        J(nnz),
        V(nnz)
    { }

    // TODO: Build from CSR/MTX

}; // struct coo_t