/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MMPR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPR_RBINARYOPERATIONS_HPP
#define MPR_RBINARYOPERATIONS_HPP

#include <operations/BinaryOperations.hpp>


/************************** COMPARISONS ****************************/


/**
 * @brief
 * R-Adapter for Checking Whether MPR Object one is Greater than MPR Object two
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RGreaterThan(DataType *apInputA, DataType *apInputB);


/**
 * @brief
 * R-Adapter for Checking Whether MPR Object is Greater than a Given Value
 *
 * @param[in] apInput
 * MPR Object
 * @param[in] aVal
 * Value to Compare MPR Values with
 *
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RGreaterThan(DataType *apInputA, double aVal);

/**
 * @brief
 * R-Adapter for Checking Whether MPR Object one is Greater than or Equal
 * MPR Object two
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RGreaterThanOrEqual(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Checking Whether MPR Object is Greater than or Equal
 * a Given Value
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aVal
 * Value to Compare MPR Values with
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RGreaterThanOrEqual(DataType *apInputA, double aVal);

/**
 * @brief
 * R-Adapter for Checking Whether MPR Object one is Less than MPR Object two
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RLessThan(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Checking Whether MPR Object is Less than a Given Value
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aVal
 * Value to Compare MPR Values with
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RLessThan(DataType *apInputA, double aVal);

/**
 * @brief
 * R-Adapter for Checking Whether MPR Object one is Less than or Equal
 * MPR Object two
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RLessThanOrEqual(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Checking Whether MPR Object is Less than or Equal a Given Value
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aVal
 * Value to Compare MPR Values with
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RLessThanOrEqual(DataType *apInputA, double aVal);

/**
 * @brief
 * R-Adapter for Checking Whether MPR Object one is Equal to MPR Object two
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
REqual(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Checking Whether MPR Object is Equal to a Given Value
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aVal
 * Value to Compare MPR Values with
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
REqual(DataType *apInputA, double aVal);

/**
 * @brief
 * R-Adapter for Checking Whether MPR Object one is Not Equal to MPR Object two
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RNotEqual(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Checking Whether MPR Object one is not Equal to a Given Value
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aVal
 * Value to Compare MPR Values with
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RNotEqual(DataType *apInputA, double aVal);


/************************** OPERATIONS ****************************/


/**
 * @brief
 * R-Adapter for Performing Plus on Two MPR Objects
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformPlus(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Performing Plus on MPR object using a Given Value
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aVal
 * Value to add to MPR Object Values
 * @param[in] aPrecision
 * Require Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformPlus(DataType *apInputA, double aVal, std::string aPrecision);

/**
 * @brief
 * R-Adapter for Performing Minus on Two MPR Objects
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformMinus(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Performing Minus with on MPR object using a Given Value
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aVal
 * Value to Minus From MPR Object Values
 * @param[in] aPrecision
 * Require Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformMinus(DataType *apInputA, double aVal, std::string aPrecision);

/**
 * @brief
 * R-Adapter for Performing Multiplication on Two MPR Objects
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformMult(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Performing Multiplication on MPR object using a Given Value
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aVal
 * Value to Multiply with on MPR Object Values
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformMult(DataType *apInputA, double aVal, std::string aPrecision);

/**
 * @brief
 * R-Adapter for Performing Division on Two MPR Objects
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformDiv(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Performing Div with on MPR object using a Given Value
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aVal
 * Value to use for Division on MPR Object Values
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformDiv(DataType *apInputA, double aVal, std::string aPrecision);

/**
 * @brief
 * R-Adapter for Performing Power Operation on Two MPR Objects
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] apInputB
 * MPR Object
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformPow(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Performing Power operation on MPR object using a Given Value
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aVal
 * Value to Perform Power operation with on MPR Object Values
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformPow(DataType *apInputA, double aVal, std::string aPrecision);

/************************** DISPATCHERS ****************************/


/**
 * @brief
 * R-Dispatcher to decide which Plus Method to use.
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aObj
 * MPR Object or Double Value ( Will throw exception otherwise)
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformPlusDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);


/**
 * @brief
 * R-Dispatcher to decide which Minus Method to use.
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aObj
 * MPR Object or Double Value ( Will throw exception otherwise)
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformMinusDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

/**
 * @brief
 * R-Dispatcher to decide which Multiply Method to use.
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aObj
 * MPR Object or Double Value ( Will throw exception otherwise)
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformMltDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

/**
 * @brief
 * R-Dispatcher to decide which Division Method to use.
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aObj
 * MPR Object or Double Value ( Will throw exception otherwise)
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformDivDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

/**
 * @brief
 * R-Dispatcher to decide which Power Method to use.
 *
 * @param[in] apInputA
 * MPR Object
 * @param[in] aObj
 * MPR Object or Double Value ( Will throw exception otherwise)
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPR Object
 *
 */
DataType *
RPerformPowDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

/************************** CONVERTERS ****************************/


/**
 * @brief
 * R-Adapter for Casting MPR Vector to R Numeric Vector
 *
 * @param[in,out] apInputA
 * MPR Object
 *
 * @returns
 * R Numeric Vector
 *
 */
std::vector <double>
RToNumericVector(DataType *apInputA);

/**
 * @brief
 * R-Adapter for Casting MPR Matrix to R Numeric Matrix
 *
 * @param[in,out] apInputA
 * MPR Object
 *
 * @returns
 * R Numeric Matrix
 *
 */
SEXP
RToNumericMatrix(DataType *apInputA);

/**
 * @brief
 * R-Adapter for Changing Precision of MPR Object
 *
 * @param[in,out] apInputA
 * MPR Object
 * @param[in] aPrecision
 * Required Output Precision
 *
 */
void
RChangePrecision(DataType *apInputA, std::string aPrecision);


#endif //MPR_RBINARYOPERATIONS_HPP
