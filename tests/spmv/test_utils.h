#pragma once

#include <iostream>
#include <string>

template<typename T>
static bool equal(T f1, T f2) { 
  return (std::fabs(f1 - f2) <= 
          std::numeric_limits<T>::epsilon() * 
          std::fmax(std::fabs(f1), std::fabs(f2)));
}

template<typename vector_t, typename index_t, typename value_t>
void cpu_spmv(csr_t<index_t, value_t>& A, vector_t x, vector_t y) {
    auto values     = A.Ax.data();
    auto indices    = A.Aj.data();
    auto offsets    = A.Ap.data();

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

template<typename vector_t>
bool validate(vector_t a, vector_t b) {
    for (size_t i = 0; i < a.size() - 1; i++) {
        if(!equal(a[i], b[i])) {
            std::cout << "i = " << i << ": " << a[i] 
                      << " != " << b[i] << std::endl;
            return false;
        }
    }

    return true;
}