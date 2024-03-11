
#ifndef MPCR_CUDAHELPERS_HPP
#define MPCR_CUDAHELPERS_HPP

#include <kernels/ContextManager.hpp>
#include <data-units/DataType.hpp>


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
                FillTriangle(DataType &aInput, const double &aValue,
                             const bool &aUpperTriangle,
                             kernels::RunContext *aContext);

            };
        }
    }
}

#endif //MPCR_CUDAHELPERS_HPP
