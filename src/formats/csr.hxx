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

    thrust::host_vector<index_t> Ap; // row offsets
    thrust::host_vector<index_t> Aj; // column indices
    thrust::host_vector<value_t> Ax; // nonzero values

    thrust::device_vector<index_t> d_Ap; // row offsets (gpu)
    thrust::device_vector<index_t> d_Aj; // column indices (gpu)
    thrust::device_vector<value_t> d_Ax; // nonzero values (gpu)

    csr_t() : num_rows(0),
              num_columns(0),
              num_nonzeros(0) {}

    csr_t(index_t r, index_t c, index_t nnz) : num_rows(r),
                                               num_columns(c),
                                               num_nonzeros(nnz),
                                               Ap(r + 1),
                                               Aj(nnz),
                                               Ax(nnz)
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

        Ap.resize(num_rows + 1);
        Aj.resize(num_nonzeros);
        Ax.resize(num_nonzeros);

        for (index_t i = 0; i < num_rows; i++)
            Ap[i] = 0;

        for (index_t i = 0; i < num_nonzeros; i++)
            Ap[rows[i]]++;

        // cumsum the nnz per row to get Bp[]
        for (index_t i = 0, cumsum = 0; i < num_rows; i++)
        {
            index_t temp = Ap[i];
            Ap[i] = cumsum;
            cumsum += temp;
        }

        Ap[num_rows] = num_nonzeros;

        // write Aj,Ax into Bj,Bx
        for (index_t i = 0; i < num_nonzeros; i++)
        {
            index_t row = rows[i];
            index_t dest = Ap[row];

            Aj[dest] = cols[i];
            Ax[dest] = data[i];

            Ap[row]++;
        }

        for (index_t i = 0, last = 0; i <= num_rows; i++)
        {
            index_t temp = Ap[i];
            Ap[i] = last;
            last = temp;
        }

        // GPU CSR
        d_Ap = Ap;
        d_Aj = Aj;
        d_Ax = Ax;
    }

}; // struct csr_t