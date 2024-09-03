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

/** Class responsible for holding the data, this class will automatically cache
 *  the data on both CPU and GPU if possible, it should provide an optimization
 *  layer when being used inside R
 **/
class DataHolder {

private:
    /** Enum describing the state of the data buffers inside the object. **/
    enum class BufferState {
        HOST_NEWER,
        DEVICE_NEWER,
        EQUAL,
        NO_HOST,
        NO_DEVICE,
        EMPTY
    };
public:

    /**
     * @brief
     * DataHolder constructor.
     * this function will create an empty DataHolder.
     *
     */
    DataHolder();

    /**
     * @brief
     * DataHolder constructor.
     * a Null pointer can be used in case of one device creation.
     *
     * @param[in] apHostPointer
     * pointer on host holding data.
     * @param[in] apDevicePointer
     * pointer on device holding the same data.
     * @param[in] aSizeInBytes
     * Size of buffers in bytes.
     *
     *
     */
    explicit
    DataHolder(char *apHostPointer, char *apDevicePointer,
               const size_t &aSizeInBytes);

    /**
     * @brief
     * DataHolder constructor.
     * create a DataHolder object with the requested buffer placement and size.
     *
     * @param[in] aSize
     * Size of buffer requested in bytes.
     * @param[in] aPlacement
     * placement enum indicating where the buffer should be allocated.
     *
     *
     */
    explicit
    DataHolder(const size_t &aSize, const OperationPlacement &aPlacement);

    /**
     * @brief
     * DataHolder de-constructor.
     *
     */
    ~DataHolder();

    /**
     * @brief
     * Data Holder copy constructor
     *
     * @param[in] aDataHolder
     * Input Data holder object to copy
     *
     * @returns
     * new DataHolder object.
     *
     */
    DataHolder(const DataHolder &aDataHolder);

    /**
     * @brief
     * DataHolder overloaded equal operator
     *
     * @param[in] aDataHolder
     * Input Data holder object to copy
     *
     * @returns
     * new DataHolder object.
     *
     */
    DataHolder &
    operator =(const DataHolder &aDataHolder);

    /**
     * @brief
     * Free a buffer according to the placement required.
     *
     *
     * @param[in] aPlacement
     * Placement enum indicating which buffer needs to be deleted.
     *
     */
    void
    FreeMemory(const OperationPlacement &aPlacement);

    /**
     * @brief
     * Allocate a buffer according to the required placement and size.
     * this function will delete any already allocated buffers.
     *
     * @param[in] aSizeInBytes
     * Size of the buffer that needs to be allocated in bytes.
     * @param[in] aPlacement
     * Placement enum
     *
     */
    void
    Allocate(const size_t &aSizeInBytes, const OperationPlacement &aPlacement);

    /**
     * @brief
     * Get buffer size in bytes.
     *
     * @returns
     * size in bytes.
     *
     */
    size_t
    GetSize();

    /**
     * @brief
     * Get data buffer according to placement required. this function will
     * automatically create/sync the buffer in case the requested buffer is not
     * created/updated.
     *
     * @param[in] aPlacement
     * Placement enum indicating which buffer should be returned.
     *
     * @returns
     * pointer to the data buffer.
     *
     */

    char *
    GetDataPointer(const OperationPlacement &aPlacement);


    /**
     * @brief
     * Set Data buffer, this function will automatically delete any existing
     * buffers on CPU and GPU,in case new buffers are passed.
     *
     * @param[in] apData
     * Data buffer pointer.
     * @param[in] aSizeInBytes
     * size of buffer in bytes.
     * @param[in] aPlacement
     * Placement enum indicating which buffer to set.
     *
     */
    void
    SetDataPointer(char *apData, const size_t &aSizeInBytes,
                   const OperationPlacement &aPlacement);

    /**
     * @brief
     * Set Data buffer, this function will automatically delete any existing
     * buffers on CPU and GPU ,in case new buffers are passed,
     * and set the the two new buffers.
     *
     * @param[in] apHostPointer
     * Data buffer pointer on host.
     * @param[in] apDevicePointer
     * Data buffer pointer on device.
     * @param[in] aSizeInBytes
     * size of buffer in bytes.
     *
     */
    void
    SetDataPointer(char *apHostPointer, char *apDevicePointer,
                   const size_t &aSizeInBytes);

    /**
     * @brief
     * Clear up and de-allocate all object data and metadata.
     *
     */
    void
    ClearUp();

    /**
     * @brief
     * Change precision of buffer. this function will detect whether to promote
     * on host or on device according to the buffer state. And in case the buffers
     * are equal, the function will decide based on the current Operation Context
     * inside the context manager.
     *
     */
    template <typename T, typename X>
    void
    ChangePrecision();


    /**
     * @brief
     * Checks if a specific buffer is allocated, according to the operation
     * placement
     *
     * @param[in] aOperationalPlacement
     * Operation placement to decide which buffer to check
     *
     * @returns
     * true if the buffer is allocated, false otherwise.
     *
     */
    inline
    bool
    IsAllocated(const OperationPlacement &aOperationalPlacement) {
        return ( aOperationalPlacement == GPU ) ? ( mpDeviceData != nullptr )
                                                : ( mpHostData != nullptr );
    };


    /**
     * @brief
     * Checks if the Data Holder is empty
     *
     * @returns
     * true if empty, false otherwise.
     *
     */
    inline
    bool
    IsEmpty() {
        return ( mpHostData == nullptr && mpDeviceData == nullptr &&
                 mSize == 0 && mBufferState == BufferState::EMPTY );
    }


private:

    /**
     * @brief
     * Sync CPU and GPU buffers, in case one is newer than the other. this function
     * will not allocate any new buffers.
     *
     */
    void
    Sync();

    /**
     * @brief
     * Sync CPU or GPU buffers according to the placement, this function
     * will not allocate any new buffers.
     *
     * @param[in] aPlacement
     * Operation placement buffer indicating which buffer to sync.
     *
     */
    void
    Sync(const OperationPlacement &aPlacement);

    /**
     * @brief
     * Change precision on host pointer, this function will automatically delete
     * the data on device.
     *
     */
    template <typename T, typename X>
    void
    PromoteOnHost();

    /**
     * @brief
     * Change precision on device pointer, this function will automatically delete
     * the data on host.
     *
     */
    template <typename T, typename X>
    void
    PromoteOnDevice();

    /**
     * @brief
     * Checks if the host or device pointer need to be allocated.
     * this function will not sync any buffers, it will only allocate.
     *
     * @param[in] aPlacement
     * check placement for buffer allocation.
     *
     */
    void
    AllocateMissingBuffer(const OperationPlacement &aPlacement);

    /**
     * @brief
     * Copy buffers and all metadata from input Data holder
     *
     * @param[in] aDataHolder
     * Data Holder to copy buffers and metadata from.
     *
     */
    void
    CopyBuffers(const DataHolder &aDataHolder);


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
