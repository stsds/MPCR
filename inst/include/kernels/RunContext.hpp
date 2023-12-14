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
#include <cstddef>


#ifdef USE_CUDA

#include <cusolverDn.h>
#include <common/Definitions.hpp>


#endif


namespace mpcr {
    namespace kernels {

        enum class RunMode {
            SYNC,
            ASYNC
        };


        class RunContext {
        public:

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
            Sync() const;


#endif

        private:
#ifdef USE_CUDA
            int *mpInfo;
            mutable void *mpWorkBuffer;
            mutable size_t mWorkBufferSize;
            cusolverDnHandle_t mCuSolverHandle;
            cudaStream_t mCudaStream;
#endif
            RunMode mRunMode;
            definitions::OperationPlacement mOperationPlacement;
        };
    }
}


#endif //MPCR_RUNCONTEXT_HPP
