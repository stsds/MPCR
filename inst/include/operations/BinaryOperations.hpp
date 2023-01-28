

#ifndef MPR_BINARYOPERATIONS_HPP
#define MPR_BINARYOPERATIONS_HPP

#include <data-units/DataType.hpp>


namespace mpr {
    namespace operations {
        namespace binary {

            template <typename T, typename X, typename Y>
            void
            PerformOperation(DataType &aInputA, DataType &aInputB,
                             DataType &aOutput, const std::string &aFun);

            template <typename T, typename X, typename Y>
            void
            PerformOperationSingle(DataType &aInputA, const double &aVal,
                                   DataType &aOutput, const std::string &aFun);


            template <typename T, typename X, typename Y>
            void
            PerformCompareOperation(DataType &aInputA, DataType &aInputB,
                                    std::vector <int> &aOutput,
                                    const std::string &aFun,
                                    Dimensions *&apDimensions);

            template <typename T>
            void
            PerformCompareOperationSingle(DataType &aInputA, const double &aVal,
                                          std::vector <int> &aOutput,
                                          const std::string &aFun,
                                          Dimensions *&apDimensions);

            void
            CheckDimensions(DataType &aInputA, DataType &aInputB);

            template <typename T, typename X, typename Y>
            void
            PerformEqualityOperation(DataType &aInputA, DataType &aInputB,
                                     std::vector <int> &aOutput,
                                     const bool &aIsNotEqual,
                                     Dimensions *&apDimensions);

            template <typename T>
            void
            PerformEqualityOperationSingle(DataType &aInputA, double &aVal,
                                           std::vector <int> &aOutput,
                                           const bool &aIsNotEqual,
                                           Dimensions *&apDimensions);

        }
    }
}


#endif //MPR_BINARYOPERATIONS_HPP
