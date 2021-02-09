
#include <thrust/device_vector.h>
#include <time.h>

#include <formats/csr.hxx>
#include <iostream>
#include <string>
#include <util/display.hxx>

#include "test_cusparse.h"
#include "test_moderngpu.h"
#include "test_utils.h"

enum SPMV_t { MGPU, CUB, CUSPARSE };

template <typename index_t = int, typename value_t = float, typename hinput_t,
          typename dinput_t, typename doutput_t>
bool run_test(SPMV_t spmv_impl, csr_t<index_t, value_t>& sparse_matrix,
              hinput_t& hin, dinput_t& din, doutput_t& dout,
              bool check = true) {
  // Reset the output vector
  thrust::fill(dout.begin(), dout.end(), 0);

  auto input_ptr = din.data();
  auto output_ptr = dout.data();

  //   Run on appropriate GPU implementation
  if (spmv_impl == MGPU) {
    // Perform moderngpu's SpMV

    spmv_mgpu(sparse_matrix, input_ptr, output_ptr);
  } else if (spmv_impl == CUB) {
  } else if (spmv_impl == CUSPARSE) {
    spmv_cusparse(sparse_matrix, input_ptr, output_ptr);
  } else {
    std::cout << "Unsupported SPMV implementation" << std::endl;
  }

  //   Copy results to CPU
  if (check) {
    thrust::host_vector<float> h_output = dout;

    //   Run on CPU
    thrust::host_vector<float> compare(sparse_matrix.num_columns);
    cpu_spmv(sparse_matrix, hin.data(), compare.data());

    //   Validate
    bool passed = validate(h_output, compare);
    if (passed) {
      std::cout << "Validation Successful" << std::endl;
      return true;
    } else {
      std::cout << "Validation Failed" << std::endl;
      return false;
    }
  }
  return true;
  //   return 0;
}

// 1) Load Matrix
// 2) Run & validate all potential algorithms (ModernGPU, cuSparse, CUB).
//      - In the future, add command line args to select only one or all
// 3) Verify (make this optional)
int main(int argc, char** argv) {
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

  //   std::cout << "Loading from Matrix Market File" << std::endl;
  sparse_matrix.build(filename);

  thrust::host_vector<float> h_input(sparse_matrix.num_columns);
  srand(time(NULL));
  for (size_t v = 0; v < h_input.size(); v++) h_input[v] = rand() % 64;

  thrust::device_vector<float> d_input = h_input;  // Only needs to occur once
  thrust::device_vector<float> d_output(sparse_matrix.num_columns);

  // ... GPU SPMV
  std::cout << "Running ModernGPU" << std::endl;
  run_test(MGPU, sparse_matrix, h_input, d_input, d_output);

  std::cout << "Running cuSparse" << std::endl;
  run_test(CUSPARSE, sparse_matrix, h_input, d_input, d_output);

  return 0;
}