/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <data-units/DataHolder.hpp>


using namespace mpcr;
using namespace mpcr::definitions;


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
                                       kernels::ContextManager::GetOperationContext());
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

    if (aPlacement == CPU) {

        if (mBufferState == BufferState::NO_DEVICE) {
            ClearUp();
            return;
        }
        this->Sync(GPU);
        memory::DestroyArray(this->mpHostData, aPlacement,
                             kernels::ContextManager::GetOperationContext());
        mBufferState = BufferState::NO_HOST;

    } else {

        if (mBufferState == BufferState::NO_HOST) {
            ClearUp();
            return;
        }
        this->Sync(CPU);
        memory::DestroyArray(this->mpDeviceData, aPlacement,
                             kernels::ContextManager::GetOperationContext());
        mBufferState = BufferState::NO_DEVICE;

    }
}


size_t
DataHolder::GetSize() {
    return this->mSize;
}


char *
DataHolder::GetDataPointer(const OperationPlacement &aPlacement) {

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
                       kernels::ContextManager::GetOperationContext(),
                       memory::MemoryTransfer::HOST_TO_DEVICE);
        mBufferState = BufferState::EQUAL;
    } else if (mBufferState == BufferState::DEVICE_NEWER) {
        // memcpy device to host
        memory::MemCpy(this->mpHostData, this->mpDeviceData, this->mSize,
                       kernels::ContextManager::GetOperationContext(),
                       memory::MemoryTransfer::DEVICE_TO_HOST);
        mBufferState = BufferState::EQUAL;
    }
}


void
DataHolder::SetDataPointer(char *apData, const size_t &aSizeInBytes,
                           const OperationPlacement &aPlacement) {

    if (apData == nullptr) {
        this->FreeMemory(aPlacement);
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


}


void
DataHolder::Sync(const OperationPlacement &aPlacement) {

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
                         kernels::ContextManager::GetOperationContext());

    memory::DestroyArray(this->mpHostData, CPU,
                         kernels::ContextManager::GetOperationContext());

    this->mpDeviceData = nullptr;
    this->mpHostData = nullptr;
    this->mSize = 0;
    this->mBufferState = BufferState::EMPTY;
}


void
DataHolder::SetDataPointer(char *apHostPointer, char *apDevicePointer,
                           const size_t &aSizeInBytes) {
    this->ClearUp();

    this->mpHostData = apHostPointer;
    this->mpDeviceData = apDevicePointer;
    this->mSize = aSizeInBytes;

    if (mSize == 0 || ( mpHostData == nullptr && mpDeviceData == nullptr )) {
        this->mBufferState = BufferState::EMPTY;
        mSize = 0;
        return;
    } else {
        if (apHostPointer == nullptr) {
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
    auto *temp = memory::AllocateArray(aSizeInBytes, aPlacement,
                                       kernels::ContextManager::GetOperationContext());
    this->SetDataPointer(temp, aSizeInBytes, aPlacement);
}


