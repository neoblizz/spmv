#pragma once

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <formats/coo.hxx>
#include <io/mtx.hxx>

/**
 * @brief Compressed Sparse Row (CSR) format.
 * 
 * @tparam index_t 
 * @tparam value_t
 */
template <typename index_t = int,
          typename value_t = float>
struct csr_t
{
    index_t num_rows;
    index_t num_columns;
    index_t num_nonzeros;

    thrust::host_vector<index_t> row_offsets; // row offsets
    thrust::host_vector<index_t> col_idx; // column indices
    thrust::host_vector<value_t> nonzero_vals; // nonzero values

    thrust::device_vector<index_t> d_row_offsets; // row offsets (gpu)
    thrust::device_vector<index_t> d_col_idx; // column indices (gpu)
    thrust::device_vector<value_t> d_nonzero_vals; // nonzero values (gpu)

    csr_t() : num_rows(0),
              num_columns(0),
              num_nonzeros(0) {}

    csr_t(index_t r, index_t c, index_t nnz) : num_rows(r),
                                               num_columns(c),
                                               num_nonzeros(nnz),
                                               row_offsets(r + 1),
                                               col_idx(nnz),
                                               nonzero_vals(nnz)
    {
    }

    void build(std::string filename)
    {
        mtx_t<index_t, value_t> mtx;
        mtx.load(filename);
        build(mtx.coo);
    }

    void build(mtx_t<index_t, value_t> &mtx)
    {
        build(mtx.coo);
    }

    void build(coo_t<index_t, value_t> &coo)
    {
        auto rows = coo.I.data();
        auto cols = coo.J.data();
        auto data = coo.V.data();

        num_rows = coo.num_rows;
        num_columns = coo.num_columns;
        num_nonzeros = coo.num_nonzeros;

        row_offsets.resize(num_rows + 1);
        col_idx.resize(num_nonzeros);
        nonzero_vals.resize(num_nonzeros);

        for (index_t i = 0; i < num_rows; i++)
            row_offsets[i] = 0;

        for (index_t i = 0; i < num_nonzeros; i++)
            row_offsets[rows[i]]++;

        // cumsum the nnz per row to get Bp[]
        for (index_t i = 0, cumsum = 0; i < num_rows; i++)
        {
            index_t temp = row_offsets[i];
            row_offsets[i] = cumsum;
            cumsum += temp;
        }

        row_offsets[num_rows] = num_nonzeros;

        // write col_idx,nonzero_vals into Bj,Bx
        for (index_t i = 0; i < num_nonzeros; i++)
        {
            index_t row = rows[i];
            index_t dest = row_offsets[row];

            col_idx[dest] = cols[i];
            nonzero_vals[dest] = data[i];

            row_offsets[row]++;
        }

        for (index_t i = 0, last = 0; i <= num_rows; i++)
        {
            index_t temp = row_offsets[i];
            row_offsets[i] = last;
            last = temp;
        }

        // GPU CSR
        d_row_offsets = row_offsets;
        d_col_idx = col_idx;
        d_nonzero_vals = nonzero_vals;
    }

}; // struct csr_t