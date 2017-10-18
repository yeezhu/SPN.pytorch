typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

int cuspn_Double_SP_InitDistanceMetric(
    THCudaDoubleTensor *distanceMetric,
    const float factor,
    const uint16 mW,
    const uint16 mH,
    const uint16 N);
int cuspn_Float_SP_InitDistanceMetric(
    THCudaTensor *distanceMetric,
    const float factor,
    const uint16 mW,
    const uint16 mH,
    const uint16 N);

int cuspn_Double_SP_Generate(
    THCudaDoubleTensor *input,
    THCudaDoubleTensor *distanceMetric,
    THCudaDoubleTensor *transferMatrix,
    THCudaDoubleTensor *proposal,
    THCudaDoubleTensor *proposalBuffer,
    const float tolerance,
    const uint32 maxIteration);
int cuspn_Float_SP_Generate(
    THCudaTensor *input,
    THCudaTensor *distanceMetric,
    THCudaTensor *transferMatrix,
    THCudaTensor *proposal,
    THCudaTensor *proposalBuffer,
    const float tolerance,
    const uint32 maxIteration);

int cuspn_Double_SP_Couple(
    THCudaDoubleTensor *input,
    THCudaDoubleTensor *proposal,
    THCudaDoubleTensor *output);
int cuspn_Float_SP_Couple(
    THCudaTensor *input,
    THCudaTensor *proposal,
    THCudaTensor *output);