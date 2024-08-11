/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/


#ifndef MPCR_CUDAMEMORYKERNELS_HPP
#define MPCR_CUDAMEMORYKERNELS_HPP

#include <kernels/ContextManager.hpp>


namespace mpcr {
    namespace kernels {

        class CudaMemoryKernels {
        public:
            /**
             * @brief
             * Template CUDA Copy function that convert one buffer from one precision to another.
             * This function will not allocate any buffer, and the memory should
             * be allocated from the outside.
             * typename T: should be source precision.
             * typename X: should be the destination precision.
             *
             * @param[in] apSource
             * Source buffer to use.
             * @param[in] apDestination
             * Destination buffer that will contain the data after copying.
             * @param[in] aNumElements
             * Number of elements inside the buffer.
             * @param[in] aContext
             * Run context containing cuda stream to use for the function.
             *
             */
            template <typename T, typename X>
            static
            void
            Copy(const T *apSource, X *apDestination, const size_t &aNumElements,
                 const kernels::RunContext *aContext);

        };


    }
}

#endif //MPCR_CUDAMEMORYKERNELS_HPP
