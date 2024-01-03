/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/


#ifndef MPCR_MEMORYHANDLER_HPP
#define MPCR_MEMORYHANDLER_HPP

#include <kernels/ContextManager.hpp>
#include <kernels/Precision.hpp>


using namespace mpcr::definitions;

namespace mpcr {
    namespace memory {

        /** Enums describing memory transfer types **/
        enum class MemoryTransfer {
            HOST_TO_DEVICE,
            DEVICE_TO_DEVICE,
            DEVICE_TO_HOST,
            HOST_TO_HOST
        };

        /**
         * @brief
         * Allocates an array of elements on the target accelerator.
         *
         *
         * @param[in] aSizeInBytes
         * Size in bytes to transfer
         *
         *
         * @return
         * A pointer to the allocated array.
         */

        char *
        AllocateArray(const size_t &aSizeInBytes,
                      const OperationPlacement &aPlacement,const kernels::RunContext *aContext);

        /**
         * @brief
         * Deallocate a previously allocated array.
         *
         * @param[in] apArray
         * The pointer to deallocate.
         *
         */
        void
        DestroyArray(char *&apArray, const OperationPlacement &aPlacement,const kernels::RunContext *aContext);

        /**
         * @brief
         * Copy memory from a source pointer to a target pointer according to the transfer type.

         * @param[in] apDestination
         * The destination pointer to copy data to.
         *
         * @param[in] apSrcDataArray
         * The source pointer to copy data from.
         *
         * @param[in] aSizeInBytes
         * Size in bytes to transfer
         *
         * @param[in] aTransferType
         * The transfer type telling the memcpy where each pointer resides(host or accelerator).
         */
        void
        MemCpy(char *&apDestination, const char *apSrcDataArray,
               const size_t &aSizeInBytes, const kernels::RunContext *aContext,
               MemoryTransfer aTransferType = MemoryTransfer::DEVICE_TO_DEVICE);

        /**
         * @brief
         * Memset mechanism for a device pointer.
         *
         *
         * @param[in] apDestination
         * The destination pointer.
         *
         * @param[in] aValue
         * The value to set each byte to.
         *
         * @param[in] aSizeInBytes
         * Size in bytes to transfer
         *
         */
        void
        Memset(char *&apDestination, char aValue, const size_t &aSizeInBytes,
               const OperationPlacement &aPlacement,
               const kernels::RunContext *aContext);

#ifdef USE_CUDA

        /** Class responsible for mapping memory handler enums to cuda enums. **/
        class MemoryDirectionConverter {
        public:

            /**
             * @brief
             * Convert Memory transfer types into cuda memcpy types
             *
             * @param[in] aTransferType
             * MemoryTransfer type
             *
             * @returns
             * Cuda Memcpy Kind enum.
             */
            inline
            static
            cudaMemcpyKind
            ToCudaMemoryTransferType(
                const memory::MemoryTransfer &aTransferType) {
                switch (aTransferType) {
                    case MemoryTransfer::HOST_TO_HOST:
                        return cudaMemcpyHostToHost;

                    case MemoryTransfer::HOST_TO_DEVICE:
                        return cudaMemcpyHostToDevice;

                    case MemoryTransfer::DEVICE_TO_DEVICE:
                        return cudaMemcpyDeviceToDevice;

                    case MemoryTransfer::DEVICE_TO_HOST:
                        return cudaMemcpyDeviceToHost;
                    default:
                        return cudaMemcpyDefault;

                }
            }
        };
#endif
    }


}


#endif //MPCR_MEMORYHANDLER_HPP
