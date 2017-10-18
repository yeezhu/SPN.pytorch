#include "libspn_kernel.h"

template <typename Dtype>
__global__ void InitDistanceMetricKernel(
    const uint32 nthreads, 
    Dtype* distanceMetric_data,
    const float factor,
    const uint16 mW,
    const uint16 mH,
    const uint16 N) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        const uint64 q = n % mW; 
        const uint64 p = (n / mW) % mH;
        const uint64 j = (n / N) % mW;
        const uint64 i = n / N / mW;
        const uint64 u = i * mW + j;
        const uint64 v = p * mW + q;
        
        if (u >= v) {
            *(distanceMetric_data + n) = expf(((i - p) * (i - p) + (j - q) * (j - q)) / (-2 * factor * factor));
            *(distanceMetric_data + v*N + u) = *(distanceMetric_data + n);
        }
    }

}

template <typename Dtype>
__global__ void InitTransferMatrixKernel(
    const uint32 nthreads, 
    const Dtype* input_data,
    const Dtype* distanceMetric_data,
    Dtype* transferMatrix_data,
    const uint16 nChannel,
    const uint16 mW,
    const uint16 mH,
    const uint16 N) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        const uint64 q = n % mW; 
        const uint64 p = (n / mW) % mH;
        const uint64 j = (n / N) % mW;
        const uint64 i = n / N / mW;
        const uint64 u = i * mW + j;
        const uint64 v = p * mW + q;

        if (i*j >= p*q) {
            uint64 c;
            float sum = 0.0f;
            for (c = 0; c < nChannel; c++) {
                const float pntA = *(input_data + c * N + i * mW + j);
                const float pntB = *(input_data + c * N + p * mW + q);
                sum += (pntA - pntB) * (pntA - pntB);
            }
            *(transferMatrix_data + n) = sqrt(sum) * *(distanceMetric_data + n);
            *(transferMatrix_data + v*N + u) = *(transferMatrix_data + n);
        }
    }
}

template <typename Dtype>
__global__ void NormTransferMatrixKernel(
    const uint32 nthreads, 
    Dtype* transferMatrix_data,
    const uint16 N) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        uint64 c;
        float sum = 0.0f;
        for (c = 0; c < N; c++) {
            sum += *(transferMatrix_data + c * N + n);
        }
        for (c = 0; c < N; c++) {
            *(transferMatrix_data + c * N + n) /= sum;
        }
    }
}

template <typename Dtype>
__global__ void UpdateProposalKernel(
    const uint32 nthreads, 
    const Dtype* src_data,
    const Dtype* diff_data,
    Dtype* dst_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        *(dst_data + n) = *(src_data + n) + *(diff_data + n);
    }
}

template <typename Dtype>
__global__ void NormProposalKernel(
    const uint32 nthreads, 
    const Dtype* src_data,
    Dtype* dst_data,
    const float scale) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        *(dst_data + n) = *(src_data + n) * scale;
    }
}

template <typename Dtype>
__global__ void CoupleKernel(
    const uint32 nthreads, 
    const Dtype* input_data,
    const Dtype* proposal_data,
    Dtype* output_data,
    const uint16 nChannel,
    const uint16 N) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        const uint64 k = n % N;
        const uint64 i = n / N / nChannel;
        *(output_data + n) = *(input_data + n) * *(proposal_data + i * N + k);
    }
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
    const uint16 N)
{
    InitDistanceMetricKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, distanceMetric_data, factor, mW, mH, N);
}
void kernel_Float_InitDistanceMetric(
    cudaStream_t stream,
    const uint32 count, 
    float* distanceMetric_data,
    const float factor,
    const uint16 mW,
    const uint16 mH,
    const uint16 N)
{
    InitDistanceMetricKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, distanceMetric_data, factor, mW, mH, N);
}

void kernel_Double_InitTransferMatrix(
    cudaStream_t stream,
    const uint32 count, 
    const double* input_data,
    const double* distanceMetric_data,
    double* transferMatrix_data,
    const uint16 nChannel,
    const uint16 mW,
    const uint16 mH,
    const uint16 N)
{
    InitTransferMatrixKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, input_data, distanceMetric_data, transferMatrix_data, nChannel, mW, mH, N);
}
void kernel_Float_InitTransferMatrix(
    cudaStream_t stream,
    const uint32 count, 
    const float* input_data,
    const float* distanceMetric_data,
    float* transferMatrix_data,
    const uint16 nChannel,
    const uint16 mW,
    const uint16 mH,
    const uint16 N)
{
    InitTransferMatrixKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, input_data, distanceMetric_data, transferMatrix_data, nChannel, mW, mH, N);
}

void kernel_Double_NormTransferMatrix(
    cudaStream_t stream,
    const uint32 count, 
    double* transferMatrix_data,
    const uint16 N)
{
    NormTransferMatrixKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, transferMatrix_data, N);
}
void kernel_Float_NormTransferMatrix(
    cudaStream_t stream,
    const uint32 count, 
    float* transferMatrix_data,
    const uint16 N)
{
    NormTransferMatrixKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, transferMatrix_data, N);
}

void kernel_Double_UpdateProposal(
    cudaStream_t stream,
    const uint32 count, 
    const double* src_data,
    const double* diff_data,
    double* dst_data)
{
    UpdateProposalKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, src_data, diff_data, dst_data);
}
void kernel_Float_UpdateProposal(
    cudaStream_t stream,
    const uint32 count, 
    const float* src_data,
    const float* diff_data,
    float* dst_data)
{
    UpdateProposalKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, src_data, diff_data, dst_data);
}

void kernel_Double_NormProposal(
    cudaStream_t stream,
    const uint32 count, 
    const double* src_data,
    double* dst_data,
    const float scale)
{
    NormProposalKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, src_data, dst_data, scale);
}
void kernel_Float_NormProposal(
    cudaStream_t stream,
    const uint32 count, 
    const float* src_data,
    float* dst_data,
    const float scale)
{
    NormProposalKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, src_data, dst_data, scale);
}

void kernel_Double_Couple(
    cudaStream_t stream,
    const uint32 count, 
    const double* input_data,
    const double* proposal_data,
    double* output_data,
    const uint16 nChannel,
    const uint16 N)
{
    CoupleKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, input_data, proposal_data, output_data, nChannel, N);
}

void kernel_Float_Couple(
    cudaStream_t stream,
    const uint32 count, 
    const float* input_data,
    const float* proposal_data,
    float* output_data,
    const uint16 nChannel,
    const uint16 N)
{
    CoupleKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, input_data, proposal_data, output_data, nChannel, N);
}

#ifdef __cplusplus
}
#endif