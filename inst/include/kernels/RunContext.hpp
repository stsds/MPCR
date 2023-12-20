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

            RunContext(const RunContext &aContext);

            ~RunContext();

            definitions::OperationPlacement
            GetOperationPlacement() const;

            void
            SetOperationPlacement(
                const definitions::OperationPlacement &aOperationPlacement);

            RunMode
            GetRunMode() const;

            void
            Sync() const;

#ifdef USE_CUDA

            cudaStream_t
            GetStream() const;

            cusolverDnHandle_t
            GetCusolverDnHandle() const;

            int *
            GetInfoPointer() const;

            void *
            RequestWorkBuffer(const size_t &aBufferSize) const;

            void
            FreeWorkBuffer()const;

        private:
            void
            ClearUp();


#endif

        private:
#ifdef USE_CUDA
            /** Integer pointer on device containing the rc values of cublas/cusolver **/
            int *mpInfo;
            /** Work buffer needed for cublas/cusolver operations **/
            mutable void *mpWorkBuffer;
            /** Work buffer size **/
            mutable size_t mWorkBufferSize;
            /** cusolver handle **/
            cusolverDnHandle_t mCuSolverHandle;
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
