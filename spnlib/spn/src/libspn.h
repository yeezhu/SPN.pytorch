typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

int spn_Double_SP_InitDistanceMetric(
    THDoubleTensor *distanceMetric,
    const float factor,
    const uint16 mW,
    const uint16 mH,
    const uint16 N);
int spn_Float_SP_InitDistanceMetric(
    THFloatTensor *distanceMetric,
    const float factor,
    const uint16 mW,
    const uint16 mH,
    const uint16 N);

int spn_Double_SP_Generate(
    THDoubleTensor *input,
    THDoubleTensor *distanceMetric,
    THDoubleTensor *transferMatrix,
    THDoubleTensor *proposal,
    THDoubleTensor *proposalBuffer,
    const float tolerance,
    const uint32 maxIteration);
int spn_Float_SP_Generate(
    THFloatTensor *input,
    THFloatTensor *distanceMetric,
    THFloatTensor *transferMatrix,
    THFloatTensor *proposal,
    THFloatTensor *proposalBuffer,
    const float tolerance,
    const uint32 maxIteration);
