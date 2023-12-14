/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_DATAHOLDER_HPP
#define MPCR_DATAHOLDER_HPP

#include <kernels/MemoryHandler.hpp>



using namespace mpcr::definitions;

class DataHolder {

private:
    enum class BufferState{
        HOST_NEWER,
        DEVICE_NEWER,
        EQUAL,
        NO_HOST,
        NO_DEVICE,
        EMPTY
    };
public:

    DataHolder();
    explicit
    DataHolder(char *apHostPointer,char*apDevicePointer,const size_t &aSizeInBytes);

    explicit
    DataHolder(const size_t &aSize, const OperationPlacement &aPlacement);

    ~DataHolder();

    void
    FreeMemory(const OperationPlacement &aPlacement);

    void
    Allocate(const size_t &aSizeInBytes,const OperationPlacement &aPlacement);

    size_t
    GetSize();

    char *
    GetDataPointer(const OperationPlacement &aPlacement);

    void
    SetDataPointer(char *apData,const size_t &aSizeInBytes,const OperationPlacement &aPlacement);

    void
    SetDataPointer(char *apHostPointer,char *apDevicePointer,const size_t &aSizeInBytes);

    void
    Sync();

    void
    Sync(const OperationPlacement &aPlacement);

    void
    ClearUp();




private:
    /** Pointer holding data in Host memory **/
    char *mpDeviceData;
    /** Pointer holding data in Device memory **/
    char *mpHostData;
    /** Total size of data in bytes **/
    size_t mSize;
    /** Enum to indicate the state of the DataHolder **/
    BufferState mBufferState;


};

#endif //MPCR_DATAHOLDER_HPP
