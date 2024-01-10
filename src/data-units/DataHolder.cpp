/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <data-units/DataHolder.hpp>
#include <utilities/MPCRDispatcher.hpp>


#ifdef USE_CUDA

#include <kernels/cuda/CudaMemoryKernels.hpp>


#endif


using namespace mpcr;
using namespace mpcr::definitions;
using namespace mpcr::kernels;


DataHolder::DataHolder() {
    mpDeviceData = nullptr;
    mpHostData = nullptr;
    mSize = 0;
    mBufferState = BufferState::EMPTY;
}


DataHolder::~DataHolder() {
    this->ClearUp();
}


DataHolder::DataHolder(const size_t &aSize,
                       const OperationPlacement &aPlacement) {
#ifndef USE_CUDA
    if(aPlacement==mpcr::definitions::GPU){
        MPCR_API_EXCEPTION("Package is compiled with no GPU support, check Operation Placement",-1);
    }
#endif

    if (aSize == 0) {
        mpDeviceData = nullptr;
        mpHostData = nullptr;
        mSize = 0;
        mBufferState = BufferState::EMPTY;
        return;
    }

    if (aPlacement == CPU) {
        mBufferState = BufferState::NO_DEVICE;
        mpDeviceData = nullptr;
    } else {
        mBufferState = BufferState::NO_HOST;
        mpHostData = nullptr;
    }

    mSize = aSize;
    auto *temp = memory::AllocateArray(aSize, aPlacement,
                                       ContextManager::GetOperationContext());
    if (aPlacement == CPU) {
        this->mpHostData = temp;
    } else {
        this->mpDeviceData = temp;
    }


}


DataHolder::DataHolder(char *apHostPointer, char *apDevicePointer,
                       const size_t &aSizeInBytes) {

    this->SetDataPointer(apHostPointer, apDevicePointer, aSizeInBytes);
}


void
DataHolder::FreeMemory(const OperationPlacement &aPlacement) {

#ifndef USE_CUDA
    if(aPlacement==mpcr::definitions::GPU && mpDeviceData!= nullptr){
        MPCR_API_EXCEPTION("Package is compiled with no GPU support, check Operation Placement",-1);
    }
#endif

    if (aPlacement == CPU) {

        if (mBufferState == BufferState::NO_DEVICE) {
            ClearUp();
            return;
        }
        this->Sync(GPU);
        memory::DestroyArray(this->mpHostData, aPlacement,
                             ContextManager::GetOperationContext());
        mBufferState = BufferState::NO_HOST;

    } else {

        if (mBufferState == BufferState::NO_HOST) {
            ClearUp();
            return;
        }
        this->Sync(CPU);
        memory::DestroyArray(this->mpDeviceData, aPlacement,
                             ContextManager::GetOperationContext());
        mBufferState = BufferState::NO_DEVICE;

    }
}


size_t
DataHolder::GetSize() {
    return this->mSize;
}


char *
DataHolder::GetDataPointer(const OperationPlacement &aPlacement) {

    if (mBufferState == BufferState::EMPTY) {
        return nullptr;
    }

    AllocateMissingBuffer(aPlacement);
    this->Sync(aPlacement);

    if (aPlacement == CPU) {
        return this->mpHostData;
    } else {
        return this->mpDeviceData;
    }
}


void
DataHolder::Sync() {
    if (mBufferState == BufferState::HOST_NEWER) {
        // memcpy host to device
        memory::MemCpy(this->mpDeviceData, this->mpHostData, this->mSize,
                       ContextManager::GetOperationContext(),
                       memory::MemoryTransfer::HOST_TO_DEVICE);
        mBufferState = BufferState::EQUAL;
    } else if (mBufferState == BufferState::DEVICE_NEWER) {
#ifndef USE_CUDA
        MPCR_API_EXCEPTION("Package is compiled with no GPU support, check Operation Placement",-1);
#endif
        // memcpy device to host
        memory::MemCpy(this->mpHostData, this->mpDeviceData, this->mSize,
                       ContextManager::GetOperationContext(),
                       memory::MemoryTransfer::DEVICE_TO_HOST);
        mBufferState = BufferState::EQUAL;
    }
}


void
DataHolder::SetDataPointer(char *apData, const size_t &aSizeInBytes,
                           const OperationPlacement &aPlacement) {
#ifndef USE_CUDA
    if(aPlacement==mpcr::definitions::GPU){
        MPCR_API_EXCEPTION("Package is compiled with no GPU support, check Operation Placement",-1);
    }
#endif


    if (apData == nullptr) {
        this->FreeMemory(aPlacement);
        return;
    }

    if (aPlacement == mpcr::definitions::GPU && apData == mpDeviceData) {
        if (mBufferState != BufferState::NO_HOST) {
            mBufferState = BufferState::DEVICE_NEWER;
        }
        return;
    } else if (aPlacement == mpcr::definitions::CPU && apData == mpHostData) {
        if (mBufferState != BufferState::NO_DEVICE) {
            mBufferState = BufferState::HOST_NEWER;
        }

        return;
    }

    this->ClearUp();

    if (aPlacement == CPU) {
        this->mpHostData = apData;
        mBufferState = BufferState::NO_DEVICE;
    } else {
        this->mpDeviceData = apData;
        mBufferState = BufferState::NO_HOST;
    }

    this->mSize = aSizeInBytes;


}


void
DataHolder::Sync(const OperationPlacement &aPlacement) {

#ifndef USE_CUDA
    if(aPlacement==mpcr::definitions::GPU){
        MPCR_API_EXCEPTION("Package is compiled with no GPU support, check Operation Placement",-1);
    }
#endif

    if (aPlacement == CPU && ( mBufferState == BufferState::HOST_NEWER ||
                               mBufferState == BufferState::NO_DEVICE ||
                               mBufferState == BufferState::EQUAL )) {
        return;
    }

    if (aPlacement == GPU && ( mBufferState == BufferState::DEVICE_NEWER ||
                               mBufferState == BufferState::NO_HOST ||
                               mBufferState == BufferState::EQUAL )) {
        return;
    }

    this->Sync();

}


void
DataHolder::ClearUp() {

    memory::DestroyArray(this->mpDeviceData, GPU,
                         ContextManager::GetOperationContext());

    memory::DestroyArray(this->mpHostData, CPU,
                         ContextManager::GetOperationContext());

    this->mpDeviceData = nullptr;
    this->mpHostData = nullptr;
    this->mSize = 0;
    this->mBufferState = BufferState::EMPTY;
}


void
DataHolder::SetDataPointer(char *apHostPointer, char *apDevicePointer,
                           const size_t &aSizeInBytes) {
    if (!( mpHostData == apHostPointer && mpDeviceData == apDevicePointer )) {
        this->ClearUp();
    }

    this->mpHostData = apHostPointer;
    this->mpDeviceData = apDevicePointer;
    this->mSize = aSizeInBytes;

    if (mSize == 0 || ( mpHostData == nullptr && mpDeviceData == nullptr )) {
        this->mBufferState = BufferState::EMPTY;
        mSize = 0;
        return;
    } else {
        if (apHostPointer == nullptr) {
#ifndef USE_CUDA
            MPCR_API_EXCEPTION("Package is compiled with no GPU support, check Operation Placement",-1);
#endif
            mBufferState = BufferState::NO_HOST;
            return;
        } else if (mpDeviceData == nullptr) {
            mBufferState = BufferState::NO_DEVICE;
            return;
        }
    }

    mBufferState = BufferState::EQUAL;
}


void
DataHolder::Allocate(const size_t &aSizeInBytes,
                     const OperationPlacement &aPlacement) {
#ifndef USE_CUDA
    if(aPlacement==mpcr::definitions::GPU){
        MPCR_API_EXCEPTION("Package is compiled with no GPU support, check Operation Placement",-1);
    }
#endif
    auto *temp = memory::AllocateArray(aSizeInBytes, aPlacement,
                                       ContextManager::GetOperationContext());
    this->SetDataPointer(temp, aSizeInBytes, aPlacement);
}


template <typename T, typename X>
void
DataHolder::ChangePrecision() {

    if (typeid(T) == typeid(X)) {
        return;
    }

    if (mBufferState == BufferState::DEVICE_NEWER ||
        mBufferState == BufferState::NO_HOST) {
        PromoteOnDevice <T, X>();

    } else if (mBufferState == BufferState::HOST_NEWER ||
               mBufferState == BufferState::NO_DEVICE) {
        PromoteOnHost <T, X>();

    } else if (mBufferState == BufferState::EQUAL) {
        if (ContextManager::GetOperationContext()->GetOperationPlacement() ==
            GPU) {
            PromoteOnDevice <T, X>();
        } else {
            PromoteOnHost <T, X>();
        }
    } else {
        return;
    }


}


template <typename T, typename X>
void
DataHolder::PromoteOnHost() {
    auto size = this->mSize / sizeof(T);
    auto pData = (T *) this->mpHostData;
    auto pData_new = (X *) memory::AllocateArray(sizeof(X) * size, CPU,
                                                 ContextManager::GetOperationContext());
    std::copy(pData, pData + size, pData_new);
    this->SetDataPointer((char *) pData_new, size * sizeof(X), CPU);
}


template <typename T, typename X>
void
DataHolder::PromoteOnDevice() {
#ifdef USE_CUDA
    auto size = this->mSize / sizeof(T);
    auto pData = (T *) this->mpDeviceData;

    bool delete_context = false;
    auto context = ContextManager::GetOperationContext();

    if (context->GetOperationPlacement() == CPU) {
        delete_context = true;
        context = new RunContext(GPU, RunMode::SYNC);
    }

    auto pData_new = (X *) memory::AllocateArray(sizeof(X) * size, GPU,
                                                 context);
    CudaMemoryKernels::Copy <T, X>(pData, pData_new, size, context);

    this->SetDataPointer((char *) pData_new, size * sizeof(X), GPU);

    if (delete_context) {
        delete context;
    }

#else
    MPCR_API_EXCEPTION("Package is compiled with no GPU support, check Placement",-1);
#endif
}


void
DataHolder::AllocateMissingBuffer(const OperationPlacement &aPlacement) {
    if (mBufferState == BufferState::NO_DEVICE &&
        aPlacement == mpcr::definitions::GPU) {
        this->mpDeviceData = memory::AllocateArray(this->mSize, GPU,
                                                   ContextManager::GetOperationContext());
        mBufferState = BufferState::HOST_NEWER;

    } else if (mBufferState == BufferState::NO_HOST &&
               aPlacement == mpcr::definitions::CPU) {
        this->mpHostData = memory::AllocateArray(this->mSize, CPU,
                                                 ContextManager::GetOperationContext());
        mBufferState = BufferState::DEVICE_NEWER;
    }
}


DataHolder::DataHolder(const DataHolder &aDataHolder) {
    this->mpHostData= nullptr;
    this->mpDeviceData= nullptr;
    this->ClearUp();
    this->CopyBuffers(aDataHolder);
}


void
DataHolder::CopyBuffers(const DataHolder &aDataHolder) {
    if (aDataHolder.mBufferState == BufferState::EMPTY) {
        return;
    } else if (aDataHolder.mBufferState == BufferState::NO_HOST ||
        aDataHolder.mBufferState == BufferState::DEVICE_NEWER) {
        this->Allocate(aDataHolder.mSize, GPU);
        memory::MemCpy(this->mpDeviceData, aDataHolder.mpDeviceData,
                       aDataHolder.mSize, ContextManager::GetOperationContext(),
                       memory::MemoryTransfer::DEVICE_TO_DEVICE);
    } else {
        this->Allocate(aDataHolder.mSize, CPU);
        memory::MemCpy(this->mpHostData, aDataHolder.mpHostData,
                       aDataHolder.mSize, ContextManager::GetOperationContext(),
                       memory::MemoryTransfer::HOST_TO_HOST);
    }

}


DataHolder &
DataHolder::operator =(const DataHolder &aDataHolder) {
    this->mpHostData= nullptr;
    this->mpDeviceData= nullptr;
    this->ClearUp();
    this->CopyBuffers(aDataHolder);
    return *this;
}


COPY_INSTANTIATE(void, DataHolder::ChangePrecision)

COPY_INSTANTIATE(void, DataHolder::PromoteOnHost)

COPY_INSTANTIATE(void, DataHolder::PromoteOnDevice)

