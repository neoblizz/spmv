
#include <thrust/device_vector.h>
#include <time.h>


#include <iostream>
#include <string>

#include <formats/csr.hxx>

#include <util/display.hxx>
#include <util/filepath.hxx>
#include <util/generate.hxx>

#include "test_cub.h"
#include "test_cusparse.h"
#include "test_moderngpu.h"
#include "test_utils.h"

enum SPMV_t { MGPU, CUB, CUSPARSE };
enum LB_t { THREAD_PER_ROW, WARP_PER_ROW, BLOCK_PER_ROW, MERGE_PATH };

template <typename index_t = int, typename value_t = float,
          typename dinput_t, typename doutput_t>
double run_test(SPMV_t spmv_impl, csr_t<index_t, value_t>& sparse_matrix,
                dinput_t& din, doutput_t& dout,
                bool check = false) {
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
  } else {
    std::cout << "Unsupported SPMV implementation" << std::endl;
  }

  //   Copy results to CPU
  if (check) {
    thrust::host_vector<float> h_output = dout;
    thrust::host_vector<float> hin = din;

    // Run on CPU
    thrust::host_vector<float> cpu_ref(sparse_matrix.num_rows);
    cpu_spmv(sparse_matrix, hin, cpu_ref);

    util::display(hin, "cpu_in");
    util::display(din, "gpu_in");
    util::display(cpu_ref, "cpu_out");
    util::display(dout, "gpu_out");

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

  if(argc <= 1 || argc >= 4) {
    std::cerr << "usage: ./bin/spmv <dataset.mtx> [--validate]" << std::endl;
    throw std::exception();
  }

  /* ========== PREPARE DATA ========== */
  bool validate = false;

  // Read in matrix market file
  std::string filename = argv[1];

  if(argc == 3)
    if(!strcmp("--validate", argv[2]))
      validate = true;

  // Construct a csr matrix from the mtx file
  csr_t<int, float> sparse_matrix;
  sparse_matrix.build(filename);
  // util::display(sparse_matrix, "sparse_matrix");

  srand(0);
  srand(time(NULL));

  thrust::device_vector<float> d_input(sparse_matrix.num_columns);
  thrust::device_vector<float> d_output(sparse_matrix.num_rows);

  // random numbers for input vector x
  util::random_uniform_distribution(d_input, 0.0f, 1.0f);

  /* ========== RUN SPMV ========== */

  double elapsed_mgpu = run_test(MGPU, sparse_matrix, d_input, d_output, validate);

  thrust::fill(thrust::device, d_output.begin(), d_output.end(), 0);
  double elapsed_cusparse = run_test(CUSPARSE, sparse_matrix, d_input, d_output, validate);

  thrust::fill(thrust::device, d_output.begin(), d_output.end(), 0);
  double elapsed_cub = run_test(CUB, sparse_matrix, d_input, d_output, validate);

  std::cout << util::extract_dataset(util::extract_filename(filename)) << "," << sparse_matrix.num_rows << ",";
  std::cout << sparse_matrix.num_columns << "," << sparse_matrix.num_nonzeros << ",";
  std::cout << elapsed_mgpu << "," << elapsed_cusparse << "," << elapsed_cub << std::endl;
}
