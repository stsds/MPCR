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

        cublasCreate(&this->mCuBlasHandle);
        cublasSetStream(this->mCuBlasHandle, this->mCudaStream);

        GPU_ERROR_CHECK(cudaMalloc((void **) &this->mpInfo, sizeof(int)));
    }
    this->mWorkBufferSizeHost = 0;
    this->mpWorkBufferHost = nullptr;
    this->mWorkBufferSizeDevice = 0;
    this->mpWorkBufferDevice = nullptr;
    this->mOperationPlacement = aOperationPlacement;

#else
    MPCR_PRINTER("Context is running without GPU support")
    MPCR_PRINTER(std::endl)
    this->mOperationPlacement = definitions::CPU ;

#endif

    this->mRunMode = aRunMode;

}


RunContext::RunContext(const RunContext &aContext) {

    this->mOperationPlacement = aContext.mOperationPlacement;
    this->mRunMode = aContext.mRunMode;


#ifdef USE_CUDA
    this->mpInfo= nullptr;
    if(this->mOperationPlacement==definitions::GPU){
        this->mCuSolverHandle = aContext.mCuSolverHandle;
        this->mCudaStream = aContext.mCudaStream;
        this->mCuBlasHandle = aContext.mCuBlasHandle;
        GPU_ERROR_CHECK(cudaMalloc((void **) &this->mpInfo, sizeof(int)));
    }
    this->mWorkBufferSizeDevice = 0;
    this->mpWorkBufferDevice = nullptr;
    this->mWorkBufferSizeHost = 0;
    this->mpWorkBufferHost = nullptr;


#else
    MPCR_PRINTER("Context is running without GPU support")
    MPCR_PRINTER(std::endl)
#endif


}


RunContext::~RunContext() {
#ifdef USE_CUDA
    int rc = 0;
    this->Sync();
    if (this->mpWorkBufferDevice != nullptr) {
        GPU_ERROR_CHECK(cudaFree(this->mpWorkBufferDevice));
    }
    if (this->mpWorkBufferHost != nullptr) {
        delete[] (char *) this->mpWorkBufferHost;
    }

    if (this->mOperationPlacement == definitions::GPU) {
        rc = cusolverDnDestroy(this->mCuSolverHandle);
        if (rc) {
            MPCR_API_EXCEPTION("Error While Destroying CuSolver Handle", rc);
        }
        rc=cublasDestroy(this->mCuBlasHandle);
        if (rc) {
            MPCR_API_EXCEPTION("Error While Destroying CuBlas Handle", rc);
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
    if (this->mOperationPlacement == definitions::GPU &&
        aOperationPlacement == definitions::GPU) {
        this->FreeWorkBufferDevice();
        return;
    }
    this->ClearUp();
    this->mOperationPlacement = aOperationPlacement;
    if (this->mOperationPlacement == definitions::GPU) {
        cudaStreamCreate(&this->mCudaStream);
        cusolverDnCreate(&this->mCuSolverHandle);
        cusolverDnSetStream(this->mCuSolverHandle, this->mCudaStream);

        cublasCreate(&this->mCuBlasHandle);
        cublasSetStream(this->mCuBlasHandle, this->mCudaStream);

        GPU_ERROR_CHECK(cudaMalloc((void **) &this->mpInfo, sizeof(int)));
    }

#else
    MPCR_PRINTER("Context is running without GPU support, Mode of operation is set automatically to CPU")
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
    this->mRunMode = aRunMode;
}

void
RunContext::FinalizeOperations(){
    if(this->mRunMode == RunMode::SYNC){
        this->Sync();
        this->FreeWorkBufferHost();
    }
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
RunContext::RequestWorkBufferDevice(const size_t &aBufferSize) const {

    if (aBufferSize == 0) {
        return this->mpWorkBufferDevice;
    }

    if (this->mOperationPlacement == definitions::CPU) {
        MPCR_API_EXCEPTION(
            "Cannot get context metadata while running CPU context", -1);
    }

    if (aBufferSize > this->mWorkBufferSizeDevice) {
        if (this->mpWorkBufferDevice != nullptr) {
            cudaFree(this->mpWorkBufferDevice);
        }
        this->mWorkBufferSizeDevice = aBufferSize;
        GPU_ERROR_CHECK(cudaMalloc(&this->mpWorkBufferDevice, aBufferSize));
    }
    return this->mpWorkBufferDevice;

}


void *
RunContext::RequestWorkBufferHost(const size_t &aBufferSize) const {

    if (aBufferSize == 0) {
        return this->mpWorkBufferHost;
    }

    if (this->mOperationPlacement == definitions::CPU) {
        MPCR_API_EXCEPTION(
            "Cannot get context metadata while running CPU context", -1);
    }

    if (aBufferSize > this->mWorkBufferSizeHost) {
        if (this->mpWorkBufferHost != nullptr) {
            delete[] (char *) this->mpWorkBufferHost;
        }
        this->mWorkBufferSizeHost = aBufferSize;
        this->mpWorkBufferHost = new char[aBufferSize];
    }
    return this->mpWorkBufferHost;

}


void RunContext::ClearUp() {
    if (this->mOperationPlacement == definitions::GPU) {
        int rc = 0;
        this->Sync();
        if (this->mpWorkBufferDevice != nullptr) {
            GPU_ERROR_CHECK(cudaFree(this->mpWorkBufferDevice));
        }
        if (this->mpWorkBufferHost != nullptr) {
            delete[] (char *) this->mpWorkBufferHost;
        }
        rc = cusolverDnDestroy(this->mCuSolverHandle);
        if (rc) {
            MPCR_API_EXCEPTION("Error While Destroying CuSolver Handle", rc);
        }
        rc=cublasDestroy(this->mCuBlasHandle);
        if (rc) {
            MPCR_API_EXCEPTION("Error While Destroying CuBlas Handle", rc);
        }
        rc = cudaStreamDestroy(this->mCudaStream);
        if (rc) {
            MPCR_API_EXCEPTION("Error While Destroying CUDA stream", rc);
        }
        GPU_ERROR_CHECK(cudaFree(this->mpInfo));
    }

    this->mpInfo = nullptr;
    this->mWorkBufferSizeDevice = 0;
    this->mpWorkBufferDevice = nullptr;
    this->mWorkBufferSizeHost = 0;
    this->mpWorkBufferHost = nullptr;
}


void
RunContext::FreeWorkBufferDevice() const {

    if (this->mOperationPlacement == definitions::GPU) {
        this->Sync();
        if (this->mpWorkBufferDevice != nullptr) {
            GPU_ERROR_CHECK(cudaFree(this->mpWorkBufferDevice));
        }
    }
    this->mWorkBufferSizeDevice = 0;
    this->mpWorkBufferDevice = nullptr;


}


void
RunContext::FreeWorkBufferHost() const {

    if (this->mOperationPlacement == definitions::GPU) {
        this->Sync();
        if (this->mpWorkBufferHost != nullptr) {
            delete[] (char *) this->mpWorkBufferHost;
        }
    }
    this->mWorkBufferSizeHost = 0;
    this->mpWorkBufferHost = nullptr;

}


cublasHandle_t
RunContext::GetCuBlasDnHandle() const {
    if (this->mOperationPlacement == definitions::CPU) {
        MPCR_API_EXCEPTION(
            "Cannot get context metadata while running CPU context", -1);
    }
    return mCuBlasHandle;
}


#endif