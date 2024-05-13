/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_CPUHELPERS_HPP
#define MPCR_CPUHELPERS_HPP

#include <kernels/ContextManager.hpp>
#include <data-units/DataType.hpp>
#include <operations/interface/Helpers.hpp>


#define MPCR_CPU_BLOCK_SIZE 8

namespace mpcr::operations {
    namespace helpers {

        template <typename T>
        class CPUHelpers : public Helpers <T> {

        public:

            CPUHelpers() = default;


            ~CPUHelpers() = default;

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

            /**
             * This Function is not implemented in CPU, since there's no
             *  use for it.
             **/
            void
            CreateIdentityMatrix(T *apData, size_t &aSideLength,
                                 kernels::RunContext *aContext);


            void
            NormMARS(DataType &aInput, T &aValue,
                     kernels::RunContext *aContext);


            void
            NormMACS(DataType &aInput, T &aValue,
                     kernels::RunContext *aContext);


            void
            NormEuclidean(DataType &aInput, T &aValue,
                          kernels::RunContext *aContext);


            void
            NormMaxMod(DataType &aInput, T &aValue,
                       kernels::RunContext *aContext);


            void
            GetRank(DataType &aInput, const double &aTolerance, T &aRank,
                    kernels::RunContext *aContext);


            void
            IsSymmetric(DataType &aInput,bool &aOutput,
                        kernels::RunContext *aContext);


            void
            CopyUpperTriangle(DataType &aInput, DataType &aOutput,
                         kernels::RunContext *aContext);
        };

        MPCR_INSTANTIATE_CLASS(CPUHelpers)
    }

}

#endif //MPCR_CPUHELPERS_HPP
