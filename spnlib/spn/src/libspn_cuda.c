#include <THC/THC.h>
#include "libspn_kernel.h"
#include "libspn_cuda.h"

#define cuspn_(NAME) TH_CONCAT_4(cuspn_, Real, _, NAME)
#define kernel_(NAME) TH_CONCAT_4(kernel_, Real, _, NAME)
#define THCUNN_assertSameGPU(...) THAssertMsg(THCudaTensor_checkGPU(__VA_ARGS__), \
  "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.")

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

#include "generic/SoftProposalGenerator.cu"
#include <THC/THCGenerateFloatType.h>

#include "generic/SoftProposalGenerator.cu"
#include <THC/THCGenerateDoubleType.h>
