/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/


#ifndef MPCR_GPUHELPERS_HPP
#define MPCR_GPUHELPERS_HPP

#include <kernels/ContextManager.hpp>
#include <data-units/DataType.hpp>
#include <operations/interface/Helpers.hpp>


#define MPCR_CUDA_BLOCK_SIZE 16

namespace mpcr::operations {
    namespace helpers {

        template <typename T>
        class GPUHelpers : public Helpers <T> {

        public:

            GPUHelpers() = default;

            ~GPUHelpers() = default;


            void
            Symmetrize(DataType &aInput, const bool &aToUpperTriangle,
                       kernels::RunContext *aContext);


            void
            Reverse(DataType &aInput, kernels::RunContext *aContext);


            void
            Transpose(DataType &aInput, kernels::RunContext *aContext);


            void
            FillTriangle(DataType &aInput, const double &aValue,
                         const bool &aUpperTriangle,
                         kernels::RunContext *aContext);

            void
            CreateIdentityMatrix(T *apData,size_t &aSideLength,kernels::RunContext *aContext);


            void
            NormMARS(DataType &aInput,T &aValue);


            void
            NormMACS(DataType &aInput,T &aValue);


            void
            NormEuclidean(DataType &aInput,T &aValue);


            void
            NormMaxMod(DataType &aInput,T &aValue);


            void
            GetRank(DataType &aInput, const double &aTolerance, T &aRank);

        };

        MPCR_INSTANTIATE_CLASS(GPUHelpers)
    }
}

#endif //MPCR_GPUHELPERS_HPP
