#pragma once

#include <stdio.h>
#include <stdlib.h>

extern "C"{
#include "mmio.h"
}

#include <util/filepath.hxx>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <formats/coo.hxx>

using mm_code_t = uint32_t;
enum : mm_code_t {
    matrix      = 1<<0,    // 0x01
    
    sparse      = 1<<1,    // 0x02
    coordinate  = 1<<2,    // 0x03
    dense       = 1<<3,    // 0x04
    array       = 1<<4,    // 0x05
    
    complex     = 1<<5,    // 0x06
    real        = 1<<6,    // 0x07
    pattern     = 1<<7,    // 0x08
    integer     = 1<<8,    // 0x09

    symmetric   = 1<<9,    // 0x10
    general     = 1<<10,   // 0x11
    skew        = 1<<11,   // 0x12
    hermitian   = 1<<12,   // 0x13

    none        = 0<<0,    // 0x00
};

template <typename index_t = int,
          typename value_t = float>
struct mtx_t {

    typedef FILE* file_t;
    typedef MM_typecode mm_type_t;

    std::string dataset;
    mm_code_t types;

    coo_t<int, float> coo;

    mtx_t() {}

    /**
     * @brief If format was specified as coordinate, 
     *        then the size line has the form:
     *          m n nonzeros
     *        where
     *          m is the number of rows in the matrix;
     *          n is the number of columns in the matrix;
     *          nonzeros is the number of nonzero entries 
     *          in the matrix (for general symmetry), or 
     *          the number of nonzero entries on or below 
     *          the diagonal (for symmetric or Hermitian 
     *          symmetry), or the number of nonzero entries 
     *          below the diagonal (for skew-symmetric symmetry).
     * 
     * @param filename 
     */
    void load(std::string filename) {
        dataset = util::extract_dataset(util::extract_filename(filename));

        file_t file;
        mm_type_t code;

        // Load MTX information
        if ((file = fopen(filename.c_str(), "r")) == NULL) {
            std::cerr << "File could not be opened: " << filename << std::endl;
            exit(1);
        }

        if (mm_read_banner(file, &code) != 0) {
            std::cerr << "Could not process Matrix Market banner" << std::endl;
            exit(1);
        }

        // Make sure we're actually reading a matrix, and not an array
        if(mm_is_array(code)) {
            std::cerr << "File is not a sparse matrix" << std::endl;
            exit(1);
        }

        index_t num_rows, num_columns, num_nonzeros;
        if ((mm_read_mtx_crd_size(file, &num_rows, &num_columns, &num_nonzeros)) !=0) {
            std::cerr << "Could not read file info (M, N, NNZ)" << std::endl;
            exit(1);
        }

        coo_t<index_t, value_t> _coo(num_rows, num_columns, num_nonzeros);

        if (mm_is_pattern(code)) {
            types |= pattern;

            // pattern matrix defines sparsity pattern, but not values
            for( index_t i = 0; i < num_nonzeros; i++ ){
                assert(fscanf(file, " %d %d \n", &(_coo.I[i]), &(_coo.J[i])) == 2);
                _coo.I[i]--;      //adjust from 1-based to 0-based indexing
                _coo.J[i]--;
                _coo.V[i] = (value_t) 1.0;  //use value 1.0 for all nonzero entries 
            }
        } else if (mm_is_real(code) || mm_is_integer(code)){
            types |= real;

            for( index_t i = 0; i < _coo.num_nonzeros; i++ ){
                index_t I, J;
                double V;  // always read in a double and convert later if necessary
                
                assert(fscanf(file, " %d %d %lf \n", &I, &J, &V) == 3);

                _coo.I[i] = (index_t) I - 1; 
                _coo.J[i] = (index_t) J - 1;
                _coo.V[i] = (value_t)  V;
            }
        } else {
            std::cerr << "Unrecognized matrix market format type" << std::endl;
            exit(1);
        }

        if(mm_is_symmetric(code)) { //duplicate off diagonal entries
            types |= symmetric;
            index_t off_diagonals = 0;
            for( index_t i = 0; i < _coo.num_nonzeros; i++ ){
                if( _coo.I[i] != _coo.J[i] )
                    off_diagonals++;
            }

            index_t _nonzeros = 2 * off_diagonals + (_coo.num_nonzeros - off_diagonals);

            thrust::host_vector<index_t> new_I(_nonzeros);
            thrust::host_vector<index_t> new_J(_nonzeros);
            thrust::host_vector<value_t> new_V(_nonzeros);

            auto _I = new_I.data();
            auto _J = new_J.data();
            auto _V = new_V.data();

            index_t ptr = 0;
            for( index_t i = 0; i < _coo.num_nonzeros; i++ ){
                if( _coo.I[i] != _coo.J[i] ){
                    _I[ptr] = _coo.I[i];  _J[ptr] = _coo.J[i];  _V[ptr] = _coo.V[i];
                    ptr++;
                    _J[ptr] = _coo.I[i];  _I[ptr] = _coo.J[i];  _V[ptr] = _coo.V[i];
                    ptr++;
                } else {
                    _I[ptr] = _coo.I[i];  new_J[ptr] = _coo.J[i];  _V[ptr] = _coo.V[i];
                    ptr++;
                }
            }
            _coo.I = new_I;  _coo.J = new_J; _coo.V = new_V;      
            _coo.num_nonzeros = _nonzeros;
        } //end symmetric case

        coo = _coo;
        fclose(file);
    }
};