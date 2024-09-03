/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_RBINARYOPERATIONS_HPP
#define MPCR_RBINARYOPERATIONS_HPP

#include <operations/BinaryOperations.hpp>


/************************** COMPARISONS ****************************/


/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object one is Greater than MPCR Object two
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RGreaterThan(DataType *apInputA, DataType *apInputB);


/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object is Greater than a Given Value
 *
 * @param[in] apInput
 * MPCR Object
 * @param[in] aVal
 * Value to Compare MPCR Values with
 *
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RGreaterThan(DataType *apInputA, double aVal);

/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object one is Greater than or Equal
 * MPCR Object two
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RGreaterThanOrEqual(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object is Greater than or Equal
 * a Given Value
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aVal
 * Value to Compare MPCR Values with
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RGreaterThanOrEqual(DataType *apInputA, double aVal);

/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object one is Less than MPCR Object two
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RLessThan(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object is Less than a Given Value
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aVal
 * Value to Compare MPCR Values with
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RLessThan(DataType *apInputA, double aVal);

/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object one is Less than or Equal
 * MPCR Object two
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RLessThanOrEqual(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object is Less than or Equal a Given Value
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aVal
 * Value to Compare MPCR Values with
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RLessThanOrEqual(DataType *apInputA, double aVal);

/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object one is Equal to MPCR Object two
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
REqual(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object is Equal to a Given Value
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aVal
 * Value to Compare MPCR Values with
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
REqual(DataType *apInputA, double aVal);

/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object one is Not Equal to MPCR Object two
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RNotEqual(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Checking Whether MPCR Object one is not Equal to a Given Value
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aVal
 * Value to Compare MPCR Values with
 * @returns
 * R-Vector/Matrix of Bool Values
 *
 */
SEXP
RNotEqual(DataType *apInputA, double aVal);


/************************** OPERATIONS ****************************/


/**
 * @brief
 * R-Adapter for Performing Plus on Two MPCR Objects
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformPlus(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Performing Plus on MPCR object using a Given Value
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aVal
 * Value to add to MPCR Object Values
 * @param[in] aPrecision
 * Require Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformPlus(DataType *apInputA, double aVal, std::string aPrecision);

/**
 * @brief
 * R-Adapter for Performing Minus on Two MPCR Objects
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformMinus(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Performing Minus with on MPCR object using a Given Value
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aVal
 * Value to Minus From MPCR Object Values
 * @param[in] aPrecision
 * Require Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformMinus(DataType *apInputA, double aVal, std::string aPrecision);

/**
 * @brief
 * R-Adapter for Performing Multiplication on Two MPCR Objects
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformMult(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Performing Multiplication on MPCR object using a Given Value
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aVal
 * Value to Multiply with on MPCR Object Values
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformMult(DataType *apInputA, double aVal, std::string aPrecision);

/**
 * @brief
 * R-Adapter for Performing Division on Two MPCR Objects
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformDiv(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Performing Div with on MPCR object using a Given Value
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aVal
 * Value to use for Division on MPCR Object Values
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformDiv(DataType *apInputA, double aVal, std::string aPrecision);

/**
 * @brief
 * R-Adapter for Performing Power Operation on Two MPCR Objects
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] apInputB
 * MPCR Object
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformPow(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R-Adapter for Performing Power operation on MPCR object using a Given Value
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aVal
 * Value to Perform Power operation with on MPCR Object Values
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPCR Object
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
 * MPCR Object
 * @param[in] aObj
 * MPCR Object or Double Value ( Will throw exception otherwise)
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformPlusDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);


/**
 * @brief
 * R-Dispatcher to decide which Minus Method to use.
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aObj
 * MPCR Object or Double Value ( Will throw exception otherwise)
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformMinusDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

/**
 * @brief
 * R-Dispatcher to decide which Multiply Method to use.
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aObj
 * MPCR Object or Double Value ( Will throw exception otherwise)
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformMltDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

/**
 * @brief
 * R-Dispatcher to decide which Division Method to use.
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aObj
 * MPCR Object or Double Value ( Will throw exception otherwise)
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformDivDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

/**
 * @brief
 * R-Dispatcher to decide which Power Method to use.
 *
 * @param[in] apInputA
 * MPCR Object
 * @param[in] aObj
 * MPCR Object or Double Value ( Will throw exception otherwise)
 * @param[in] aPrecision
 * Required Output Precision (should be greater than or equal to the input precision)
 * @returns
 * MPCR Object
 *
 */
DataType *
RPerformPowDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

/************************** CONVERTERS ****************************/


/**
 * @brief
 * R-Adapter for Casting MPCR Vector to R Numeric Vector
 *
 * @param[in,out] apInputA
 * MPCR Object
 *
 * @returns
 * R Numeric Vector
 *
 */
std::vector <double>
RToNumericVector(DataType *apInputA);

/**
 * @brief
 * R-Adapter for Casting MPCR Matrix to R Numeric Matrix
 *
 * @param[in,out] apInputA
 * MPCR Object
 *
 * @returns
 * R Numeric Matrix
 *
 */
SEXP
RToNumericMatrix(DataType *apInputA);

/**
 * @brief
 * R-Adapter for Changing Precision of MPCR Object
 *
 * @param[in,out] apInputA
 * MPCR Object
 * @param[in] aPrecision
 * Required Output Precision
 *
 */
void
RChangePrecision(DataType *apInputA, std::string aPrecision);


#endif //MPCR_RBINARYOPERATIONS_HPP
