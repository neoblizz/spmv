#include <moderngpu/kernel_segreduce.hxx>

// using namespace mgpu;
template<typename launch_arg_t = mgpu::empty_t,
            typename index_t = int, typename value_t = float,
            typename input_t, typename output_t>
void spmv_mgpu(csr_t<index_t, value_t>& A, input_t& input, output_t& output) {

    // ... GPU SPMV
    // GPU device context
    // mgpu::standard_context_t context;
    
    // auto values   = A.d_Ax.data();
    // auto indices  = A.d_Aj.data();
    // auto offsets  = A.d_Ap.data();
    
    // int offsets_size = A.num_rows;
    // int nnz = A.num_nonzeros;

    // mgpu::spmv(values, indices, input, nnz, offsets, offsets_size, output, context);

    // // Synchronize the device
    // context.synchronize();
}