
#ifndef MPR_LINEARALGEBRA_HPP
#define MPR_LINEARALGEBRA_HPP


#include <data-units/DataType.hpp>

#define LAYOUT blas::Layout::ColMajor


namespace mpr {
    namespace operations {
        namespace linear {

            template <typename T>
            void
            CrossProduct(DataType &aInputA, DataType &aInputB,
                         DataType &aOutput,const bool &aTranspose=false);

            template<typename T>
            void
            IsSymmetric(DataType &aInput,bool &aOutput);


        }
    }
}

#endif //MPR_LINEARALGEBRA_HPP
