/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MMPR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPR_BINARYOPERATIONS_HPP
#define MPR_BINARYOPERATIONS_HPP

#include <data-units/DataType.hpp>


namespace mpr {
    namespace operations {
        namespace binary {


            /**
             * @brief
             * Perform Operation on 2 MPR Objects according to provided Operation
             * aFun.
             *
             * @param[in] aInputA
             * MPR object can be Vector or Matrix
             * @param[out] aInputB
             * MPR object can be Vector or Matrix
             * @param[out] aOutput
             * MPR Object can be a vector or a Matrix according to the given inputs
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
             * Perform Operation on MPR Object and a Numerical Value
             * according to provided Operation aFun.
             *
             * @param[in] aInputA
             * MPR object can be Vector or Matrix
             * @param[out] aVal
             * Numerical Value
             * @param[out] aOutput
             * MPR Object can be a vector or a Matrix according to the given inputs
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
             * Perform Compare Operation on  two MPR Object according to
             * provided Operation aFun.
             *
             * @param[in] aInputA
             * MPR object can be Vector or Matrix
             * @param[out] aInputB
             * MPR object can be Vector or Matrix
             * @param[out] aOutput
             * MPR Object can be a vector or a Matrix according to the given inputs
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
             * Perform Compare Operation on MPR Object and a Numerical Value
             * according to  provided Operation aFun.
             *
             * @param[in] aInputA
             * MPR object can be Vector or Matrix
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
             * Function to check whether the input MPR objects Dimensions match
             * the required Dimensions , it will throw an error incase operation
             * cannot be performed on th given Dimensions.
             *
             * @param[in] aInputA
             * MPR object can be Vector or Matrix
             * @param[out] aInputB
             * MPR object can be Vector or Matrix
             *
             */
            void
            CheckDimensions(DataType &aInputA, DataType &aInputB);

            /**
             * @brief
             * Perform Equality Operation on two MPR Object
             *
             * @param[in] aInputA
             * MPR object can be Vector or Matrix
             * @param[out] aInputB
             * MPR object can be Vector or Matrix
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
             * Perform Equality Operation on MPR Object and a Numerical Value
             *
             * @param[in] aInputA
             * MPR object can be Vector or Matrix
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


#endif //MPR_BINARYOPERATIONS_HPP
