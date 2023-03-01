

#ifndef MPR_RMATHEMATICALOPERATIONS_HPP
#define MPR_RMATHEMATICALOPERATIONS_HPP

#include <operations/MathematicalOperations.hpp>


/**
 * @brief
 * Perform Abs operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RAbs(DataType *aInput);

/**
 * @brief
 * Perform Sqrt operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RSqrt(DataType *aInput);

/**
 * @brief
 * Perform Ceil operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RCeiling(DataType *aInput);

/**
 * @brief
 * Perform Floor operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RFloor(DataType *aInput);

/**
 * @brief
 * Perform Truncate operation on MPR Object.
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RTruncate(DataType *aInput);

/**
 * @brief
 * Perform Round operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @param[in] aDecimalPlaces
 * number of decimal places used for rounding.
 * default will remove all decimal points
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RRound(DataType *aInput, const int &aDecimalPlaces);

/**
 * @brief
 * Perform exp(x) operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RExp(DataType *aInput);

/**
 * @brief
 * Perform exp (x) -1 operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RExp1m(DataType *aInput);

/**
 * @brief
 * Perform tgamma operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RGamma(DataType *aInput);

/**
 * @brief
 * Perform lgamma operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RLGamma(DataType *aInput);

/**
 * @brief
 * Check if elements is finite or not.
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * Logical Vector/Matrix according to Input True if finite ,false otherwise.
 *
 */
SEXP
RIsFinite(DataType *aInput);

/**
 * @brief
 * Check if elements is infinite or not.
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * Logical Vector/Matrix according to Input True if infinite ,false finite ,
 * and NAN.
 *
 */
SEXP
RIsInFinite(DataType *aInput);

/**
 * @brief
 * Check if elements is NAN or not.
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * Logical Vector/Matrix according to Input True if NAN ,false Otherwise.
 *
 */
SEXP
RIsNan(DataType *aInput);

/**
 * @brief
 * Perform Log operation on MPR Object default 1.
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @param[in] aBase
 * 1 :log->exp(1)  2: log2   10: log10
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RLog(DataType *aInput, int aBase);

/**
 * @brief
 * Perform Log Base 10 operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RLog10(DataType *aInput);

/**
 * @brief
 * Perform Log Base 2 operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RLog2(DataType *aInput);

/**
 * @brief
 * Perform Sin operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RSin(DataType *aInput);

/**
 * @brief
 * Perform Cos operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RCos(DataType *aInput);

/**
 * @brief
 * Perform Tan operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RTan(DataType *aInput);

/**
 * @brief
 * Perform aSin operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RASin(DataType *aInput);

/**
 * @brief
 * Perform aCos operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RACos(DataType *aInput);

/**
 * @brief
 * Perform aTan operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RATan(DataType *aInput);

/**
 * @brief
 * Perform Sinh operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RSinh(DataType *aInput);

/**
 * @brief
 * Perform Cosh operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RCosh(DataType *aInput);

/**
 * @brief
 * Perform Tanh operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RTanh(DataType *aInput);

/**
 * @brief
 * Perform aSinh operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RASinh(DataType *aInput);

/**
 * @brief
 * Perform aCosh operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RACosh(DataType *aInput);

/**
 * @brief
 * Perform aTanh operation on MPR Object
 *
 * @param[in] aInputA
 * MPR object can be Vector or Matrix
 * @returns
 * MPR Object can be a vector or a Matrix according to the given inputs
 *
 */
DataType *
RATanh(DataType *aInput);


#endif //MPR_RMATHEMATICALOPERATIONS_HPP
