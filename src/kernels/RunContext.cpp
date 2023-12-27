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
    this->mpInfo = nullptr;
    if (aOperationPlacement == definitions::GPU) {
        cudaStreamCreate(&this->mCudaStream);
        cusolverDnCreate(&this->mCuSolverHandle);
        cusolverDnSetStream(this->mCuSolverHandle, this->mCudaStream);
        GPU_ERROR_CHECK(cudaMalloc((void **) &this->mpInfo, sizeof(int)));
    }
    this->mWorkBufferSize = 0;
    this->mpWorkBuffer = nullptr;
    this->mOperationPlacement = aOperationPlacement;

#else
    MPCR_PRINTER("Context is running without GPU support")
    MPCR_PRINTER(std::endl)
    this->mOperationPlacement = definitions::CPU ;

#endif

    this->mRunMode = aRunMode;

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
    if (this->mOperationPlacement == definitions::GPU) {
        rc = cusolverDnDestroy(this->mCuSolverHandle);
        if (rc) {
            MPCR_API_EXCEPTION("Error While Destroying CuSolver Handle", rc);
        }
        rc = cudaStreamDestroy(this->mCudaStream);
        if (rc) {
            MPCR_API_EXCEPTION("Error While Destroying CUDA stream", rc);
        }
        GPU_ERROR_CHECK(cudaFree(this->mpInfo));
    }

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
    this->ClearUp();
    this->mOperationPlacement = aOperationPlacement;
    if (this->mOperationPlacement == definitions::GPU) {
        cudaStreamCreate(&this->mCudaStream);
        cusolverDnCreate(&this->mCuSolverHandle);
        cusolverDnSetStream(this->mCuSolverHandle, this->mCudaStream);
        GPU_ERROR_CHECK(cudaMalloc((void **) &this->mpInfo, sizeof(int)));
    }

#else
    MPCR_PRINTER("\"Context is running without GPU support, Mode of operation is set automatically to CPU")
    MPCR_PRINTER(std::endl)
    this->mOperationPlacement=definitions::CPU;
#endif
}


void
RunContext::Sync() const {
#ifdef USE_CUDA
    if (this->mOperationPlacement == definitions::GPU) {
        cudaStreamSynchronize(this->mCudaStream);
    }
#endif
}


void
RunContext::SetRunMode(const RunMode &aRunMode) {
    if (this->mRunMode == RunMode::ASYNC && aRunMode == RunMode::SYNC) {
        this->Sync();
    }
    this->mRunMode=aRunMode;
}

/** -------------------------- CUDA code -------------------------- **/

#ifdef USE_CUDA


cudaStream_t
RunContext::GetStream() const {
    if (this->mOperationPlacement == definitions::CPU) {
        MPCR_API_EXCEPTION(
            "Cannot get context metadata while running CPU context", -1);
    }
    return this->mCudaStream;
}


cusolverDnHandle_t
RunContext::GetCusolverDnHandle() const {
    if (this->mOperationPlacement == definitions::CPU) {
        MPCR_API_EXCEPTION(
            "Cannot get context metadata while running CPU context", -1);
    }
    return this->mCuSolverHandle;
}


int *
RunContext::GetInfoPointer() const {
    if (this->mOperationPlacement == definitions::CPU) {
        MPCR_API_EXCEPTION(
            "Cannot get context metadata while running CPU context", -1);
    }
    return this->mpInfo;
}


void *
RunContext::RequestWorkBuffer(const size_t &aBufferSize) const {

    if (this->mOperationPlacement == definitions::CPU) {
        MPCR_API_EXCEPTION(
            "Cannot get context metadata while running CPU context", -1);
    }

    if (aBufferSize > this->mWorkBufferSize) {
        if (this->mpWorkBuffer != nullptr) {
            cudaFree(this->mpWorkBuffer);
        }
        this->mWorkBufferSize = aBufferSize;
        GPU_ERROR_CHECK(cudaMalloc(&this->mpWorkBuffer, aBufferSize));
    }
    return this->mpWorkBuffer;

}


void RunContext::ClearUp() {
    if (this->mOperationPlacement == definitions::GPU) {
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
    }

    this->mpInfo = nullptr;
    this->mWorkBufferSize = 0;
    this->mpWorkBuffer = nullptr;
}


void
RunContext::FreeWorkBuffer() const {

    if (this->mOperationPlacement == definitions::GPU) {
        this->Sync();
        if (this->mpWorkBuffer != nullptr) {
            GPU_ERROR_CHECK(cudaFree(this->mpWorkBuffer));
        }
    }
    this->mWorkBufferSize = 0;
    this->mpWorkBuffer = nullptr;

}



#endif