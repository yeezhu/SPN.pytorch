#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SoftProposalGenerator.cu"
#else

int cuspn_(SP_InitDistanceMetric)(
    THCTensor *distanceMetric,
    const float factor,
    const uint16 mW,
    const uint16 mH,
    const uint16 N)
{
    const uint16 count = N * N;

    THCTensor_(resize2d)(state, distanceMetric, N, N);
    real *distanceMetric_data = THCTensor_(data)(state, distanceMetric);

    kernel_(InitDistanceMetric)(
        THCState_getCurrentStream(state), 
        count, 
        distanceMetric_data, 
        factor, 
        mW,
        mH,
        N);
    THCudaCheck(cudaGetLastError());

    return 1;
}

int cuspn_(SP_Generate)(
    THCTensor *input,
    THCTensor *distanceMetric,
    THCTensor *transferMatrix,
    THCTensor *proposal,
    THCTensor *proposalBuffer,
    const float tolerance,
    const uint32 maxIteration)
{
    uint8 ndim = input->nDimension;
    THArgCheck(ndim == 4, 1, "only supports batch mode.");

    const uint16 nBatch = input->size[0];
    const uint16 nChannel = input->size[1];
    const uint16 mW = input->size[2];
    const uint16 mH = input->size[3];
    const uint16 N = mW * mH;
    const uint32 nEntry = nChannel * N;

    THCUNN_assertSameGPU(state, 5, input, distanceMetric, transferMatrix, proposal, proposalBuffer);
    THCTensor_(resizeAs)(state, transferMatrix, distanceMetric);
    THCTensor_(resize3d)(state, proposal, nBatch, mW, mH);
    THCTensor_(resize2d)(state, proposalBuffer, mW, mH);

    // (nBatch, nChannel, mH, mW)
    real *input_data = THCTensor_(data)(state, input);
    // (mH*mW, mH*mW)
    real *distanceMetric_data = THCTensor_(data)(state, distanceMetric);
    // (mH*mW, mH*mW)
    real *transferMatrix_data = THCTensor_(data)(state, transferMatrix);
    // (nBatch, mH, mW)
    real *proposal_data = THCTensor_(data)(state, proposal);
    // (mH, mW)
    real *proposalBuffer_data = THCTensor_(data)(state, proposalBuffer);

    const float avg = 1.0f / N;
    float sumOver;
    float sumOver_debug;
    uint32 count;
    uint32 i, j;

    THCTensor_(fill)(state, proposal, avg);

    for (i = 0; i < nBatch; i++) {
        /* init transfer matrix */
        count = N * N;
        kernel_(InitTransferMatrix)(
            THCState_getCurrentStream(state), 
            count, 
            input_data + i * nEntry, 
            distanceMetric_data, 
            transferMatrix_data, 
            nChannel, 
            mW, 
            mH,
            N);
        THCudaCheck(cudaGetLastError());

        count = N;
        kernel_(NormTransferMatrix)(
            THCState_getCurrentStream(state), 
            count,
            transferMatrix_data, 
            N);
        THCudaCheck(cudaGetLastError());
        
        /* generate soft proposal for each sample */
        // init buffer
        THCTensor_(fill)(state, proposalBuffer, avg);
        for (j = 0; j < maxIteration; j++) {
            // calculate diffs
            THCudaBlas_Sgemv(
                state,
                't',
                N,
                N,
                1.0f,
                transferMatrix_data,
                N,
                proposal_data,
                1,
                -1.0f,
                proposalBuffer_data,
                1);

            float normDiff = THCTensor_(normall)(state, proposalBuffer, 2);
            if (normDiff < tolerance) break;
            // add diffs
            kernel_(UpdateProposal)(
                THCState_getCurrentStream(state), 
                count,
                proposal_data, 
                proposalBuffer_data,
                proposalBuffer_data);
            THCudaCheck(cudaGetLastError());

            sumOver = THCTensor_(sumall)(state, proposalBuffer);
            if (sumOver < 0) break;
            // norm proposal
            kernel_(NormProposal)(
                THCState_getCurrentStream(state), 
                count,
                proposalBuffer_data, 
                proposal_data,
                1.0f / sumOver);
            
            THCudaCheck(cudaGetLastError());
        }
        proposal_data += N;
    }
    return 1;
}

int cuspn_(SP_Couple)(
    THCTensor *input,
    THCTensor *proposal,
    THCTensor *output)
{
    uint8 ndim = input->nDimension;
    THArgCheck(ndim == 4, 1, "only supports batch mode.");
    
    THCUNN_assertSameGPU(state, 3, input, proposal, output);

    const uint16 nBatch = input->size[0];
    const uint16 nChannel = input->size[1];
    const uint16 mW = input->size[2];
    const uint16 mH = input->size[3];
    const uint16 N = mW * mH;

    real *input_data = THCTensor_(data)(state, input);
    // (nBatch, mH, mW)
    real *proposal_data = THCTensor_(data)(state, proposal);
    real *output_data = THCTensor_(data)(state, output);

    const uint32 count = nBatch * nChannel * N;

    kernel_(Couple)(
        THCState_getCurrentStream(state), 
        count,
        input_data, 
        proposal_data,
        output_data,
        nChannel,
        N);
    THCudaCheck(cudaGetLastError());    

    return 1;
}

#endif