/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/


#include <kernels/MemoryHandler.hpp>


#ifdef USE_CUDA

#include <kernels/cuda/CudaMemoryKernels.hpp>
#include <utilities/MPCRDispatcher.hpp>


#endif


using namespace mpcr;
using namespace memory;


char *
memory::AllocateArray(const size_t &aSizeInBytes,
                      const OperationPlacement &aPlacement,
                      const kernels::RunContext *aContext) {

    char *pdata = nullptr;

    if (aSizeInBytes == 0) {
        return pdata;
    }

#ifdef USE_CUDA
    if (aPlacement == definitions::GPU) {
        GPU_ERROR_CHECK(cudaMalloc((void **) &pdata, aSizeInBytes));
    }
#endif
    if (aPlacement == definitions::CPU) {
        pdata = new char[aSizeInBytes];
    }

    return pdata;
}


void
memory::DestroyArray(char *&apArray, const OperationPlacement &aPlacement,
                     const kernels::RunContext *aContext) {

    if (apArray != nullptr) {
#ifdef USE_CUDA
        if (aPlacement == definitions::GPU) {
            GPU_ERROR_CHECK(cudaFree(apArray));
        }
#endif
        if (aPlacement == definitions::CPU) {
            delete[]apArray;
        }
    }
    apArray = nullptr;

}


void
memory::MemCpy(char *apDestination, const char *apSrcDataArray,
               const size_t &aSizeInBytes, const kernels::RunContext *aContext,
               MemoryTransfer aTransferType) {
    if (aSizeInBytes == 0) {
        return;
    }
#ifdef USE_CUDA
    if (aTransferType != MemoryTransfer::HOST_TO_HOST) {
        if (aContext == nullptr ||
            aContext->GetOperationPlacement() == definitions::CPU) {
            MPCR_API_EXCEPTION(
                "CUDA Memcpy cannot be performed with CPU context", -1);
        }
        GPU_ERROR_CHECK(
            cudaMemcpyAsync(apDestination, apSrcDataArray,
                            aSizeInBytes,
                            MemoryDirectionConverter::ToCudaMemoryTransferType(
                                aTransferType),
                            aContext->GetStream()));
        if (aContext->GetRunMode() == kernels::RunMode::ASYNC) {
            cudaStreamSynchronize(aContext->GetStream());
        }
    }
#endif

    if (aTransferType == MemoryTransfer::HOST_TO_HOST) {
        memcpy(apDestination, apSrcDataArray, aSizeInBytes);
    }

}


void
memory::Memset(char *apDestination, char aValue, const size_t &aSizeInBytes,
               const OperationPlacement &aPlacement,
               const kernels::RunContext *aContext) {
#ifdef USE_CUDA
    if (aPlacement == definitions::GPU) {
        if (aContext == nullptr ||
            aContext->GetOperationPlacement() == definitions::CPU) {
            MPCR_API_EXCEPTION(
                "CUDA Memcpy cannot be performed with CPU context", -1);
        }
        GPU_ERROR_CHECK(cudaMemsetAsync(apDestination, aValue, aSizeInBytes,
                                        aContext->GetStream()))
        if (aContext->GetRunMode() == kernels::RunMode::ASYNC) {
            cudaStreamSynchronize(aContext->GetStream());
        }
    }
#endif

    if (aPlacement == definitions::CPU) {
        memset(apDestination, aValue, aSizeInBytes);
    }
}


#ifdef USE_CUDA


template <typename T, typename X>
void
memory::CopyDevice(const char *apSource, char *apDestination,
                   const size_t &aNumElements) {

    auto pData_src = (T *) apSource;
    auto pData_des = (X *) apDestination;
    auto context = kernels::ContextManager::GetOperationContext();
    kernels::CudaMemoryKernels::Copy <T, X>(pData_src, pData_des, aNumElements,
                                            context);

}


COPY_INSTANTIATE(void, memory::CopyDevice, const char *apSource,
                 char *apDestination, const size_t &aNumElements)


#endif
