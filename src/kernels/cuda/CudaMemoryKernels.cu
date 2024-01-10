/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <kernels/cuda/CudaMemoryKernels.hpp>
#include <utilities/MPCRDispatcher.hpp>


using namespace mpcr::kernels;


template <typename T, typename X>
__global__
void
PerformCopy(const T *apSource, X *apDestination,
            size_t aNumElements) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < aNumElements) {
        apDestination[ tid ] = static_cast<X>(apSource[ tid ]);
    }
}


template <typename T, typename X>
void
CudaMemoryKernels::Copy(const T *apSource, X *apDestination,
                        const size_t &aNumElements,
                        const kernels::RunContext *aContext) {
    auto threadsPerBlock = 256;
    auto blocksPerGrid =
        ( aNumElements + threadsPerBlock - 1 ) / threadsPerBlock;

    PerformCopy <T, X><<<blocksPerGrid,
    threadsPerBlock, 0, aContext->GetStream()>>>
        (apSource, apDestination, aNumElements);

    aContext->Sync();


}



COPY_INSTANTIATE_ONE(void,CudaMemoryKernels::Copy,const size_t &aNumElements,
                     const kernels::RunContext *aContext)

