/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_BINARYOPERATIONS_HPP
#define MPCR_BINARYOPERATIONS_HPP

#include <data-units/DataType.hpp>


namespace mpcr {
    namespace operations {
        namespace binary {


            /**
             * @brief
             * Perform Operation on 2 MPCR Objects according to provided Operation
             * aFun.
             *
             * @param[in] aInputA
             * MPCR object can be Vector or Matrix
             * @param[out] aInputB
             * MPCR object can be Vector or Matrix
             * @param[out] aOutput
             * MPCR Object can be a vector or a Matrix according to the given inputs
             * @param[in] aFun
             * string indicating which operation to perform
             * currently supported ( + , - , * , / , ^ )
             *
             */
            template <typename T, typename X, typename Y>
            void
            PerformOperation(DataType &aInputA, DataType &aInputB,
                             DataType &aOutput, const std::string &aFun);

            /**
             * @brief
             * Perform Operation on MPCR Object and a Numerical Value
             * according to provided Operation aFun.
             *
             * @param[in] aInputA
             * MPCR object can be Vector or Matrix
             * @param[out] aVal
             * Numerical Value
             * @param[out] aOutput
             * MPCR Object can be a vector or a Matrix according to the given inputs
             * @param[in] aFun
             * string indicating which operation to perform
             * currently supported ( + , - , * , / , ^ )
             *
             */
            template <typename T, typename X, typename Y>
            void
            PerformOperationSingle(DataType &aInputA, const double &aVal,
                                   DataType &aOutput, const std::string &aFun);


            /**
             * @brief
             * Perform Compare Operation on  two MPCR Object according to
             * provided Operation aFun.
             *
             * @param[in] aInputA
             * MPCR object can be Vector or Matrix
             * @param[out] aInputB
             * MPCR object can be Vector or Matrix
             * @param[out] aOutput
             * MPCR Object can be a vector or a Matrix according to the given inputs
             * @param[in] aFun
             * string indicating which operation to perform
             * currently supported ( > , >= , < , <= )
             * @param[out] apDimensions
             * Dimensions to e set incase the output is a Matrix .This Will change
             * the output shape incase it's not returned with nullptr
             *
             */
            template <typename T, typename X, typename Y>
            void
            PerformCompareOperation(DataType &aInputA, DataType &aInputB,
                                    std::vector <int> &aOutput,
                                    const std::string &aFun,
                                    Dimensions *&apDimensions);

            /**
             * @brief
             * Perform Compare Operation on MPCR Object and a Numerical Value
             * according to  provided Operation aFun.
             *
             * @param[in] aInputA
             * MPCR object can be Vector or Matrix
             * @param[out] aVal
             * Numeric Value to use for Comparison
             * @param[out] aOutput
             * vector of int values  1/TRUE 0/FALSE INT_MIN/NA
             * @param[in] aFun
             * string indicating which operation to perform
             * currently supported ( > , >= , < , <= )
             * @param[out] apDimensions
             * Dimensions to e set incase the output is a Matrix .This Will change
             * the output shape incase it's not returned with nullptr
             *
             */
            template <typename T>
            void
            PerformCompareOperationSingle(DataType &aInputA, const double &aVal,
                                          std::vector <int> &aOutput,
                                          const std::string &aFun,
                                          Dimensions *&apDimensions);

            /**
             * @brief
             * Function to check whether the input MPCR objects Dimensions match
             * the required Dimensions , it will throw an error incase operation
             * cannot be performed on th given Dimensions.
             *
             * @param[in] aInputA
             * MPCR object can be Vector or Matrix
             * @param[out] aInputB
             * MPCR object can be Vector or Matrix
             *
             */
            void
            CheckDimensions(DataType &aInputA, DataType &aInputB);

            /**
             * @brief
             * Perform Equality Operation on two MPCR Object
             *
             * @param[in] aInputA
             * MPCR object can be Vector or Matrix
             * @param[out] aInputB
             * MPCR object can be Vector or Matrix
             * @param[out] aOutput
             * vector of int values  1/TRUE 0/FALSE INT_MIN/NA
             * @param[in] aIsNotEqual
             * Flag to indicate which operation should be performed !=/True or ==/False
             * @param[out] apDimensions
             * Dimensions to e set incase the output is a Matrix .This Will change
             * the output shape incase it's not returned with nullptr
             *
             */
            template <typename T, typename X, typename Y>
            void
            PerformEqualityOperation(DataType &aInputA, DataType &aInputB,
                                     std::vector <int> &aOutput,
                                     const bool &aIsNotEqual,
                                     Dimensions *&apDimensions);

            /**
             * @brief
             * Perform Equality Operation on MPCR Object and a Numerical Value
             *
             * @param[in] aInputA
             * MPCR object can be Vector or Matrix
             * @param[out] aVal
             * Numeric Value to use for comparison
             * @param[out] aOutput
             * vector of int values  1/TRUE 0/FALSE INT_MIN/NA
             * @param[in] aIsNotEqual
             * Flag to indicate which operation should be performed !=/True or ==/False
             * @param[out] apDimensions
             * Dimensions to e set incase the output is a Matrix .This Will change
             * the output shape incase it's not returned with nullptr
             *
             */
            template <typename T>
            void
            PerformEqualityOperationSingle(DataType &aInputA, double &aVal,
                                           std::vector <int> &aOutput,
                                           const bool &aIsNotEqual,
                                           Dimensions *&apDimensions);

        }
    }
}


#endif //MPCR_BINARYOPERATIONS_HPP
