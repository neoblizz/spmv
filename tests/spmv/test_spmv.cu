
#include <thrust/device_vector.h>
#include <time.h>

#include <formats/csr.hxx>
#include <iostream>
#include <string>
#include <util/display.hxx>

#include "test_cub.h"
#include "test_cusparse.h"
#include "test_moderngpu.h"
#include "test_tiled.h"
#include "test_utils.h"

enum SPMV_t { MGPU, CUB, CUSPARSE, TILED };

template <typename index_t = int, typename value_t = float, typename hinput_t,
          typename dinput_t, typename doutput_t>
double run_test(SPMV_t spmv_impl, csr_t<index_t, value_t>& sparse_matrix,
                hinput_t& hin, dinput_t& din, doutput_t& dout,
                bool check = true) {
  // Reset the output vector
  thrust::fill(dout.begin(), dout.end(), 0);

  double elapsed_time = 0;

  //   Run on appropriate GPU implementation
  if (spmv_impl == MGPU) {
    elapsed_time = spmv_mgpu(sparse_matrix, din, dout);
  } else if (spmv_impl == CUB) {
    elapsed_time = spmv_cub(sparse_matrix, din, dout);
  } else if (spmv_impl == CUSPARSE) {
    elapsed_time = spmv_cusparse(sparse_matrix, din, dout);
  } else if (spmv_impl == TILED) {
    elapsed_time = spmv_tiled(sparse_matrix, din, dout);
  } else {
    std::cout << "Unsupported SPMV implementation" << std::endl;
  }

  printf("GPU finished in %lf ms\n", elapsed_time);

  //   Copy results to CPU
  if (check) {
    thrust::host_vector<float> h_output = dout;
    util::display(h_output, "h_output");

    // Run on CPU
    thrust::host_vector<float> cpu_ref(sparse_matrix.num_rows);
    cpu_spmv(sparse_matrix, hin, cpu_ref);

    for (index_t row = 0; row < sparse_matrix.num_rows; row++) {
      cpu_ref[row] = 0.0;
      // Loop over all the non-zeroes within A's row
      for (auto k = sparse_matrix.row_offsets[row]; k < sparse_matrix.row_offsets[row + 1]; ++k)
        cpu_ref[row] += sparse_matrix.nonzero_vals[k] * hin[sparse_matrix.col_idx[k]];
    }

    util::display(hin, "host_in");
    util::display(din, "gpu_in");
    util::display(dout, "gpu_out");
    util::display(cpu_ref, "cpu_out");

    // Validate
    bool passed = validate(h_output, cpu_ref);
    if (passed) {
      std::cout << "Validation Successful" << std::endl;
      return elapsed_time;
    } else {
      std::cout << "Validation Failed" << std::endl;
      return -1;
    }
  }
  return elapsed_time;
}

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

  util::display(sparse_matrix, "sparse_matrix");

  thrust::host_vector<float> h_input(sparse_matrix.num_columns);
  srand(0);
  // srand(time(NULL));
  for (size_t v = 0; v < h_input.size(); v++) h_input[v] = rand() % 64;

  thrust::device_vector<float> d_input = h_input;  // Only needs to occur once
  thrust::device_vector<float> d_output(sparse_matrix.num_rows);

  // GPU SPMV
  // std::cout << "Running ModernGPU" << std::endl;
  // double elapsed_mgpu =
  //     run_test(MGPU, sparse_matrix, h_input, d_input, d_output);

  std::cout << "Running cuSparse" << std::endl;
  double elapsed_cusparse =
      run_test(CUSPARSE, sparse_matrix, h_input, d_input, d_output);

  // std::cout << "Running CUB" << std::endl;
  // double elapsed_cub = run_test(CUB, sparse_matrix, h_input, d_input,
  // d_output);

  // std::cout << "Running Tiled" << std::endl;
  // double elapsed_tiled =
  //     run_test(TILED, sparse_matrix, h_input, d_input, d_output);

  printf("%s,%d,%d,%d,%f\n", filename.c_str(), sparse_matrix.num_rows,
         sparse_matrix.num_columns, sparse_matrix.num_nonzeros,
         elapsed_cusparse);

  return 0;
}