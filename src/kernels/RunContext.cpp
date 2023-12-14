/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/


#include <kernels/RunContext.hpp>


using namespace mpcr::kernels;
using namespace mpcr;


RunContext::RunContext(
    const definitions::OperationPlacement &aOperationPlacement,
    const RunMode &aRunMode) {
#ifdef USE_CUDA

    cudaStreamCreate(&this->mCudaStream);
    cusolverDnCreate(&this->mCuSolverHandle);
    cusolverDnSetStream(this->mCuSolverHandle, this->mCudaStream);
    GPU_ERROR_CHECK(cudaMalloc((void **) &this->mpInfo, sizeof(int)));
    this->mWorkBufferSize = 0;
    this->mpWorkBuffer = nullptr;


#else
    MPCR_PRINTER("Context is running without GPU support")
    MPCR_PRINTER(std::endl)
#endif

    this->mRunMode = aRunMode;
    this->mOperationPlacement = aOperationPlacement;
}


RunContext::RunContext(const RunContext &aContext) {

#ifdef USE_CUDA
    this->mCuSolverHandle = aContext.mCuSolverHandle;
    this->mCudaStream = aContext.mCudaStream;
    this->mpInfo = aContext.mpInfo;
    this->mWorkBufferSize = 0;
    this->mpWorkBuffer = nullptr;


#else
    MPCR_PRINTER("Context is running without GPU support")
    MPCR_PRINTER(std::endl)
#endif

    this->mOperationPlacement = aContext.mOperationPlacement;
    this->mRunMode = aContext.mRunMode;

}


RunContext::~RunContext() {
#ifdef USE_CUDA
    int rc = 0;
    this->Sync();
    if (this->mpWorkBuffer != nullptr) {
        GPU_ERROR_CHECK(cudaFree(this->mpWorkBuffer));
    }
    rc = cusolverDnDestroy(this->mCuSolverHandle);
    if (rc) {
        MPCR_API_EXCEPTION("Error While Destroying CuSolver Handle", rc);
    }
    rc = cudaStreamDestroy(this->mCudaStream);
    if (rc) {
        MPCR_API_EXCEPTION("Error While Destroying CUDA stream", rc);
    }
    GPU_ERROR_CHECK(cudaFree(this->mpInfo));
#endif
}


RunMode
RunContext::GetRunMode() const {
    return this->mRunMode;
}


definitions::OperationPlacement
RunContext::GetOperationPlacement() const {
    return this->mOperationPlacement;
}


void
RunContext::SetOperationPlacement(
    const definitions::OperationPlacement &aOperationPlacement) {
#ifdef USE_CUDA

    this->mOperationPlacement = aOperationPlacement;


#else
    MPCR_PRINTER("\"Context is running without GPU support, Mode of operation is set automatically to CPU")
    MPCR_PRINTER(std::endl)
    this->mOperationPlacement=definitions::CPU;
#endif
}


#ifdef USE_CUDA


cudaStream_t
RunContext::GetStream() const {
    return this->mCudaStream;
}


cusolverDnHandle_t
RunContext::GetCusolverDnHandle() const {
    return this->mCuSolverHandle;
}


int *
RunContext::GetInfoPointer() const {
    return this->mpInfo;
}


void *
RunContext::RequestWorkBuffer(const size_t &aBufferSize) const {

    if (aBufferSize > this->mWorkBufferSize) {
        if (this->mpWorkBuffer != nullptr) {
            cudaFree(this->mpWorkBuffer);
        }
        this->mWorkBufferSize = aBufferSize;
        GPU_ERROR_CHECK(cudaMalloc(&this->mpWorkBuffer, aBufferSize));
    }
    return this->mpWorkBuffer;

}


void
RunContext::Sync() const {
    cudaStreamSynchronize(this->mCudaStream);
}


#endif