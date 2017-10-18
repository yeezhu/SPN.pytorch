typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;
 
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#ifdef __cplusplus
extern "C" {
#endif

void kernel_Double_InitDistanceMetric(
    cudaStream_t stream,
    const uint32 count, 
    double* distanceMetric_data,
    const float factor,
    const uint16 mW,
    const uint16 mH,
    const uint16 N);
void kernel_Float_InitDistanceMetric(
    cudaStream_t stream,
    const uint32 count, 
    float* distanceMetric_data,
    const float factor,
    const uint16 mW,
    const uint16 mH,
    const uint16 N);

void kernel_Double_InitTransferMatrix(
    cudaStream_t stream,
    const uint32 count, 
    const double* input_data,
    const double* distanceMetric_data,
    double* transferMatrix_data,
    const uint16 nChannel,
    const uint16 mW,
    const uint16 mH,
    const uint16 N);
void kernel_Float_InitTransferMatrix(
    cudaStream_t stream,
    const uint32 count, 
    const float* input_data,
    const float* distanceMetric_data,
    float* transferMatrix_data,
    const uint16 nChannel,
    const uint16 mW,
    const uint16 mH,
    const uint16 N);

void kernel_Double_NormTransferMatrix(
    cudaStream_t stream,
    const uint32 count, 
    double* transferMatrix_data,
    const uint16 N);
void kernel_Float_NormTransferMatrix(
    cudaStream_t stream,
    const uint32 count, 
    float* transferMatrix_data,
    const uint16 N);

void kernel_Double_UpdateProposal(
    cudaStream_t stream,
    const uint32 count, 
    const double* src_data,
    const double* diff_data,
    double* dst_data);
void kernel_Float_UpdateProposal(
    cudaStream_t stream,
    const uint32 count, 
    const float* src_data,
    const float* diff_data,
    float* dst_data);

void kernel_Double_NormProposal(
    cudaStream_t stream,
    const uint32 count, 
    const double* src_data,
    double* dst_data,
    const float scale);
void kernel_Float_NormProposal(
    cudaStream_t stream,
    const uint32 count, 
    const float* src_data,
    float* dst_data,
    const float scale);

void kernel_Double_Couple(
    cudaStream_t stream,
    const uint32 count, 
    const double* input_data,
    const double* proposal_data,
    double* output_data,
    const uint16 nChannel,
    const uint16 N);
void kernel_Float_Couple(
    cudaStream_t stream,
    const uint32 count, 
    const float* input_data,
    const float* proposal_data,
    float* output_data,
    const uint16 nChannel,
    const uint16 N);

#ifdef __cplusplus
}
#endif