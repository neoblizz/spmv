#include <iostream>
#include <string>
#include <time.h>

#include <moderngpu/kernel_segreduce.hxx>
#include <thrust/device_vector.h>

#include <util/display.hxx>
#include <formats/csr.hxx>

template<typename T>
static bool equal(T f1, T f2) { 
  return (std::fabs(f1 - f2) <= 
          std::numeric_limits<T>::epsilon() * 
          std::fmax(std::fabs(f1), std::fabs(f2)));
}

using namespace mgpu;
template<typename launch_arg_t = empty_t,
            typename index_t = int, typename value_t = float,
            typename vector_t, typename output_t>
void spmv(csr_t<index_t, value_t>& A, vector_t& x, output_t& output, context_t& context) {
    
    auto values   = A.d_Ax.data();
    auto indices  = A.d_Aj.data();
    auto offsets  = A.d_Ap.data();
    
    int offsets_size = A.num_rows;
    int nnz = A.num_nonzeros;

    spmv(values, indices, x, nnz, offsets, offsets_size, output, context);
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

int main(int argc, char ** argv) {
    
    // ... PREPARE DATA
    // Read in matrix market file
    std::string filename;

    if (argc == 2) {
        filename = argv[1];
    } else {
        std::cerr << "Usage: /path/to/bin/csr filename.mtx" << std::endl;
        exit(1);
    }

    // Construct a csr matrix from the mtx file
    csr_t<int, float> sparse_matrix;

    std::cout << "Loading from Matrix Market File" << std::endl;
    sparse_matrix.build(filename);

    thrust::host_vector<float> vector(sparse_matrix.num_columns);
    srand(time(NULL));
    for (size_t v = 0; v < vector.size(); v++)
        vector[v] = rand() % 64;

    thrust::device_vector<float> d_vector = vector;
    thrust::device_vector<float> d_output(sparse_matrix.num_columns);

    // ... GPU SPMV
    // GPU device context
    mgpu::standard_context_t context;

    // Perform moderngpu's SpMV
    auto input  = d_vector.data();
    auto output = d_output.data();

    spmv(sparse_matrix, input, output, context);

    // Synchronize the device
    context.synchronize();

    // Copy results to CPU
    thrust::host_vector<float> h_output = d_output;

    // ... CPU SPMV
    thrust::host_vector<float> compare(sparse_matrix.num_columns);
    cpu_spmv(sparse_matrix, vector.data(), compare.data());

    // ... VALIDATE
    bool passed = validate(h_output, compare);
    if (passed)
        std::cout << "Validation Successful" << std::endl;
    else
        std::cout << "Validation Failed" << std::endl;

    // Print Output
    util::display(sparse_matrix,    "sparse matrix");
    util::display(vector,           "input vector");
    util::display(h_output,         "gpu output");
    util::display(compare,          "cpu output");

    return 0;
}