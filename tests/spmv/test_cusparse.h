#include <cusparse.h>         // cusparseSpMV

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


template<typename launch_arg_t = mgpu::empty_t,
            typename index_t = int, typename value_t = float,
            typename input_t, typename output_t>
void spmv_cusparse(csr_t<index_t, value_t>& A, input_t& input, output_t& output) {
}