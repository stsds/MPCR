/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/


#ifndef MPCR_CUDAHELPERS_HPP
#define MPCR_CUDAHELPERS_HPP

#include <kernels/ContextManager.hpp>
#include <data-units/DataType.hpp>

#define MPCR_CUDA_BLOCK_SIZE 16

namespace mpcr {
    namespace operations {
        namespace helpers {
            class CudaHelpers {

            public:
                template <typename T>
                static
                void
                Symmetrize(DataType &aInput, const bool &aToUpperTriangle,
                           kernels::RunContext *aContext);

                template <typename T>
                static
                void
                Reverse(DataType &aInput, kernels::RunContext *aContext);

                template <typename T>
                static
                void
                Transpose(DataType &aInput, kernels::RunContext *aContext);

                template <typename T>
                static
                void
                FillTriangle(DataType &aInput, const double &aValue,
                             const bool &aUpperTriangle,
                             kernels::RunContext *aContext);

            };
        }
    }
}

#endif //MPCR_CUDAHELPERS_HPP
