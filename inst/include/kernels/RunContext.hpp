/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_RUNCONTEXT_HPP
#define MPCR_RUNCONTEXT_HPP

#include <utilities/MPCRErrorHandler.hpp>
#include <common/Definitions.hpp>
#include <cstddef>


#ifdef USE_CUDA

#include <cusolverDn.h>


#endif


namespace mpcr {
    namespace kernels {

        /** Enum describing the cuda stream behavior (async,sync),
         * not used in the case of CPU Context.  **/
        enum class RunMode {
            SYNC,
            ASYNC
        };


        class RunContext {
        public:
            /**
             * @brief
             * Run Context constructor
             *
             * @param[in] aOperationPlacement
             * Enum indicating whether the stream is CPU or GPU.
             * @param[in] aRunMode
             * Run mode indicating whether the stream is sync or async.
             *
             */
            explicit
            RunContext(
                const definitions::OperationPlacement &aOperationPlacement = definitions::CPU,
                const RunMode &aRunMode = RunMode::SYNC);

            /**
             * @brief
             * Run Context copy constructor.
             * This Copy constructor will not allocate a work buffer it only
             * copies the metadata
             *
             * @param[in] aContext
             * RunContext to create a new Context from.
             *
             */
            RunContext(const RunContext &aContext);

            /**
             * @brief
             * Run Context destructor.
             */
            ~RunContext();

            /**
             * @brief
             * Get the Operation placement, indicating whether the context is
             * for GPU or CPU.
             *
             * GPU context can work for both CPU & GPU operations, however,
             * the CPU context can't
             *
             * @returns
             * Operation placement.
             */
            definitions::OperationPlacement
            GetOperationPlacement() const;

            /**
             * @brief
             * Set the Operation placement, to indicate whether the context is
             * for GPU or CPU.
             *
             * GPU context can work for both CPU & GPU operations, however,
             * the CPU context can't
             *
             * @param[in] aOperationPlacement
             * Operation placement enum CPU,GPU.
             *
             */
            void
            SetOperationPlacement(
                const definitions::OperationPlacement &aOperationPlacement);

            /**
             * @brief
             * Get the RunMode for the context, indicating whether the context is
             * SYNC or ASYNC, useful only in the case of GPU
             *
             * @returns
             * Run Mode
             */
            RunMode
            GetRunMode() const;

            /**
             * @brief
             * Set the RunMode for the context, to indicate whether the context is
             * SYNC or ASYNC, useful only in the case of GPU
             *
             * @param[in] aRunMode
             * Run Mode enum indicating whether the context is SYNC or ASYNC
             */
            void
            SetRunMode(const RunMode &aRunMode);

            /**
             * @brief
             * sync the context CUDA stream.
             *
             */
            void
            Sync() const;

            /**
             * @brief
             * Cleans up and synchronizes resources in SYNC mode.
             *
             * Frees the host work buffer and syncs.
             */
            void
            FinalizeOperations();

            /**
             * @brief
             * Cleans up and synchronizes resources.
             *
             * Frees the host work buffer and syncs.
             */
            void
            FinalizeRunContext();

#ifdef USE_CUDA

            /**
             * @brief
             * Get context CUDA stream to be used for any operation.
             *
             * @returns
             * CUDA stream.
             *
             */
            cudaStream_t
            GetStream() const;

            /**
             * @brief
             * Get context CuSolver handle.
             *
             * @returns
             * CuSolver handle.
             *
             */
            cusolverDnHandle_t
            GetCusolverDnHandle() const;

            /**
             * @brief
             * Get context CuBlas handle.
             *
             * @returns
             * Cublas handle.
             *
             */
            cublasHandle_t
            GetCuBlasDnHandle() const;

            /**
             * @brief
             * Get Information pointer used as output for any CuSolver call.
             *
             * @returns
             * info pointer on GPU
             *
             */
            int *
            GetInfoPointer() const;

            /**
             * @brief
             * Request a GPU work buffer for CuSolver/CuBlas operations.
             * The function will allocate a buffer in case the buffer size requested
             * is larger than the ine already allocated, if not, it will return
             * the allocated work buffer.
             *
             * @returns
             * void pointer to the allocated work buffer.
             *
             */
            void *
            RequestWorkBufferDevice(const size_t &aBufferSize) const;

            /**
             * @brief
             * Request a CPU work buffer for CuSolver/CuBlas operations.
             * The function will allocate a buffer in case the buffer size requested
             * is larger than the ine already allocated, if not, it will return
             * the allocated work buffer.
             *
             * @returns
             * void pointer to the allocated work buffer.
             *
             */
            void *
            RequestWorkBufferHost(const size_t &aBufferSize) const;

            /**
             * @brief
             * Sync the stream and then free the allocated work buffer.
             *
             */
            void
            FreeWorkBufferDevice() const;

            /**
             * @brief
             * Sync the stream and then free the allocated work buffer.
             *
             */
            void
            FreeWorkBufferHost() const;

        private:

            /**
             * @brief
             * Clear up all the allocated memory and destroys all the handles.
             * THis function doesn't change the state of context, so the RunContext
             * and the Operation placement will not be changed.
             *
             */
            void
            ClearUp();


#endif

        private:
#ifdef USE_CUDA
            /** Integer pointer on device containing the rc values of cublas/cusolver **/
            int *mpInfo;
            /** GPU Work buffer needed for cublas/cusolver operations **/
            mutable void *mpWorkBufferDevice;
            /** CPU Work buffer needed for cublas/cusolver operations **/
            mutable void *mpWorkBufferHost;
            /** Work buffer size **/
            mutable size_t mWorkBufferSizeDevice;
            /** Work buffer size **/
            mutable size_t mWorkBufferSizeHost;
            /** cusolver handle **/
            cusolverDnHandle_t mCuSolverHandle;
            /** cublas handle **/
            cublasHandle_t  mCuBlasHandle;
            /** Cuda stream **/
            cudaStream_t mCudaStream;
#endif
            /** Enum indicating whether the operation is sync or async **/
            RunMode mRunMode;
            /** Enum indicating whether the operation is done on GPU or CPU **/
            definitions::OperationPlacement mOperationPlacement;
        };
    }
}


#endif //MPCR_RUNCONTEXT_HPP