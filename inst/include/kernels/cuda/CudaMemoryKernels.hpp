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

            template <typename T, typename X>
            static
            void
            Copy(const T *apSource, X *apDestination, const size_t &aNumElements,
                 const kernels::RunContext *aContext);

        };


    }
}

#endif //MPCR_CUDAMEMORYKERNELS_HPP
