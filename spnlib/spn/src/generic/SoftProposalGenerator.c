#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftProposalGenerator.c"
#else

int spn_(SP_InitDistanceMetric)(
    THTensor *distanceMetric,
    const float factor,
    const uint16 mW,
    const uint16 mH,
    const uint16 N)
{
    printf("cpu version is not implemented.\n");
    return 1;
}

int spn_(SP_Generate)(
    THTensor *input,
    THTensor *distanceMetric,
    THTensor *transferMatrix,
    THTensor *proposal,
    THTensor *proposalBuffer,
    const float tolerance,
    const uint32 maxIteration)
{
    printf("cpu version is not implemented.\n");
    return 1;
}

#endif
