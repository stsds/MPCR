/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_RBASICUTILITIES_HPP
#define MPCR_RBASICUTILITIES_HPP

#include <data-units/DataType.hpp>


/**
 * @brief
 * R Adapter for Combining two Matrices by columns
 *
 * @param[in] apInputA
 * MPCR Matrix one
 * @param[in] apInputB
 * MPCR Matrix two
 *
 * @return
 * MPCR Matrix holding combined data
 *
 */
DataType *
RCBind(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R Adapter for Combining two Matrices by rows
 *
 * @param[in] apInputA
 * MPCR Matrix one
 * @param[in] apInputB
 * MPCR Matrix two
 *
 * @return
 * MPCR Matrix holding combined data
 *
 */
DataType *
RRBind(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R Adapter for Checking if MPCR object is 16-bit Precision
 *
 * @param[in] apInput
 * MPCR Object
 *
 * @returns
 * True if the object is holding 16-bit precision object,
 * false otherwise
 *
 */
bool
RIsSFloat(DataType *apInput);

/**
 * @brief
 * R Adapter for Checking if MPCR object is 32-bit Precision
 *
 * @param[in] apInput
 * MPCR Object
 *
 * @returns
 * True if the object is holding 32-bit precision object,
 * false otherwise
 *
 */
bool
RIsFloat(DataType *apInput);

/**
 * @brief
 * R Adapter for Checking if MPCR object is 64-bit Precision
 *
 * @param[in] apInput
 * MPCR Object
 *
 * @returns
 * True if the object is holding 64-bit precision object,
 * false otherwise
 *
 */
bool
RIsDouble(DataType *apInput);

/**
 * @brief
 * R Adapter for Replicating value(s) number of times
 * in-case aLength =0 the output size will be size*input size ,else size=aLength
 *
 * @param[in] apInput
 * MPCR object to replicate
 * @param[in] aSize
 * Size of output vector
 * @param[in] aLength
 * Length of Output Value
 *
 * @returns
 * MPCR Vector Holding Replicated Data
 *
 */
DataType *
RReplicate(DataType *apInput, size_t aSize, size_t aLength);

/**
 * @brief
 * R Adapter for Removing NA values from Vector.
 *
 * @param[in,out] apInput
 * MPCR Object.
 *
 */
void
RNaExclude(DataType *apInput);

/**
 * @brief
 * R Adapter for Replacing NA values with a given value.
 *
 * @param[in,out] apInput
 * MPCR Object.
 * @param[in] aValue
 * Value to use it instead of NA's
 *
 */
void
RNaReplace(DataType *apInput, double aValue);

/**
 * @brief
 * R adapter for Getting Diagonal of a Matrix.
 * MPCR Object must be a Matrix
 *
 * @param[in] apInput
 * MPCR Matrix
 *
 * @returns
 * MPCR Object holding Diagonals
 *
 */
DataType *
RGetDiagonal(DataType *apInput);

/**
 * @brief
 * R adapter for Getting Diagonal of a Matrix.
 * MPCR can be a Vector , but dims must be passed
 *
 * @param[in] apInput
 * MPCR object can be Vector or Matrix
 * @param[out] aRow
 * Number of Rows
 * @param[in] aCol
 * Number of Cols
 *
 * @returns
 * MPCR Object holding Diagonals
 *
 */
DataType *
RGetDiagonalWithDims(DataType *apInput, size_t aRow, size_t aCol);

/**
 * @brief
 * R Adapter for Printing string indicating whether it's 16/32/64 Bit Precision
 *
 * @param[in] apInput
 * MPCR object can be Vector or Matrix
 *
 */
void
RGetType(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Min Element in Array
 *
 * @param[in] apInput
 * MPCR object can be Vector or Matrix
 *
 * @returns
 * MPCR Object holding Minimum Val (Same Precision)
 *
 */
DataType *
RGetMin(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Min Element Index in Array
 *
 * @param[in] apInput
 * MPCR object can be Vector or Matrix
 *
 * @returns
 * Index of Min Element
 *
 */
size_t
RGetMinIdx(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Max Element in Array
 *
 * @param[in] apInput
 * MPCR object can be Vector or Matrix
 *
 * @returns
 * MPCR Object holding Maximum Val (Same Precision)
 *
 */
DataType *
RGetMax(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Max Element Index in Array
 *
 * @param[in] apInput
 * MPCR object can be Vector or Matrix
 *
 * @returns
 * Index of Max Element
 *
 */
size_t
RGetMaxIdx(DataType *apInput);

/**
 * @brief
 * R Adapter for Applying operation (+,-,*,/) to the row or column in Matrix.
 *
 * @param[in] apInput
 * MPCR Matrix
 * @param[in] apStats
 * the value(s) that should be used in the operation
 * @param[in] aMargin
 * aMargin = 1 means row; aMargin = otherwise means column.
 * @param[in] aOperation
 * char containing operation (+,-,*,/)
 *
 * @returns
 * MPCR Vector holding Data After Applying Sweep
 *
 */
DataType *
RSweep(DataType *apInput, DataType *apStats, int aMargin,
       std::string aOperation);

/**
 * @brief
 * R Adapter for Checking Whether Element at index is NAN or Not
 *
 * @param[in] apInput
 * MPCR Object
 * @param[in] aIndex
 * Index of Element to check
 *
 * @returns
 * true if NAN,-NAN else Otherwise
 *
 */
SEXP
RIsNa(DataType *apInput, long aIdx);

/**
 * @brief
 * Get total size of Memory used by MPCR Object
 *
 * @param[in] apInput
 * MPCR Object
 *
 * @returns
 * Total size of Memory used by MPCR Object
 *
 */
size_t
RObjectSize(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Number of Rows
 *
 * @param[in] apInput
 * MPCR Object
 *
 * @returns
 * Number of Rows in a Matrix
 */
size_t
RGetNRow(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Number of Columns
 *
 * @param[in] apInput
 * MPCR Object
 *
 * @returns
 * Number of Rows in a Matrix
 */

size_t
RGetNCol(DataType *apInput);

/**
 * @brief
 * R Adapter for Printing Information about MPCR object
 * Dimensions-Matrix/Vector-Values ,and Precisions.
 *
 * @param[in] apInput
 * MPCR Object.
 *
 */
void
RPrint(DataType *apInput);

/**
 * @brief
 * R Adapter for Concatenating List of MPCR Vectors into one MPCR Vector.
 * This Function Casts the SEXP pointer to DataTypes pointers , And Check a Magic
 * Number inside the MPCR Class to determine if its a MPCR object or Not.
 *
 * Warning:
 * There's a very Small Possibility that the Passed Objects' Magic Number is
 * the Same as DataType , in this case , The behavior of the function is unexpected.
 * So the User should check whether all Objects are MPCR Objects or not.
 *
 * @param[in] aList
 * List Of SEXP
 *
 * @returns
 * MPCR Vector containing all values in all lists (Precision = Highest Precision
 * in the List)
 *
 */
DataType *
RConcatenate(Rcpp::ListOf <SEXP> aList);

/**
 * @brief
 *  R Adapter for Centering and/or Scaling the columns of a numeric matrix.
 *
 * @param[in] apInputA
 * MPCR Object.
 * @param[in] apCenter
 * numeric-alike MPCR vector of length equal to the number of
 * columns of aInput.
 * @param[out] apScale
 * numeric-alike MPCR vector of length equal to the number of
 * columns of aInput.
 * @returns
 * MPCR Object with the same size and shape after centering and/or scaling.
 *
 */
DataType *
RScale(DataType *apInput, DataType *apCenter, DataType *apScale);

/**
 * @brief
 *  R Adapter for Centering and/or Scaling the columns of a numeric matrix.
 *
 * @param[in] apInputA
 * MPCR Object.
 * @param[in] aCenter
 * bool if true centering is done using column mean ,else no centering is done.
 * @param[out] apScale
 * numeric-alike MPCR vector of length equal to the number of
 * columns of aInput.
 * @returns
 * MPCR Object with the same size and shape after centering and/or scaling.
 *
 */
DataType *
RScale(DataType *apInput, bool aCenter, DataType *apScale);

/**
 * @brief
 *  R Adapter for Centering and/or Scaling the columns of a numeric matrix.
 *
 * @param[in] apInputA
 * MPCR Object.
 * @param[in] apCenter
 * numeric-alike MPCR vector of length equal to the number of
 * columns of aInput.
 * @param[out] aScale
 * bool if true scaling is done using column standard deviation,else no scaling
 * is done.
 * @returns
 * MPCR Object with the same size and shape after centering and/or scaling.
 *
 */
DataType *
RScale(DataType *apInput, DataType *apCenter, bool aScale);

/**
 * @brief
 *  R Adapter for Centering and/or Scaling the columns of a numeric matrix.
 *
 * @param[in] apInputA
 * MPCR Object.
 * @param[in] aCenter
 * bool if true centering is done using column mean ,else no centering is done.
 * @param[out] aScale
 * bool if true scaling is done using column standard deviation,else no scaling
 * is done.
 * @returns
 * MPCR Object with the same size and shape after centering and/or scaling.
 *
 */
DataType *
RScale(DataType *apInput, bool aCenter, bool aScale);

/**
 * @brief
 *  R Adapter for Centering and/or Scaling the columns of a numeric matrix.
 *  Centering is done using column mean and scaling is done using column standard
 *  deviation.
 *
 * @param[in] apInputA
 * MPCR Object.
 * @returns
 * MPCR Object with the same size and shape after centering and/or scaling.
 *
 */
DataType *
RScale(DataType *apInput);

/**
 * @brief
 *  R Adapter for Dispatching Rscale
 *
 * @param[in] SEXP
 * SEXP Object.
 * @param[in] SEXP
 * SEXP Object.
 * @param[in] SEXP
 * SEXP Object.

 * @returns
 * MPCR Object with the same size and shape after centering and/or scaling.
 *
 */
DataType *
RScaleDispatcher(SEXP a, SEXP b, SEXP c);

/**
 * @brief
 *  Converts R vector or Matrix to MPCR object.
 *  if aRow or aCol = zero , MPCR vector will be created , else MPCR Matrix.
 *
 * @param[in] aValues
 * R vector/Matrix holding values to create MPCR object from.
 * @param[in] aRow
 * Number of Rows in case of creating an MPCR Matrix .
 * @param[in] aCol
 * Number of Cols in case of creating an MPCR Matrix .
 * @param[in] aPrecision
 * Required Precision of the created MPCR Object.
 *
 * @returns
 * New MPCR Object constructed from the given inputs
 *
 */
DataType *
RConvertToMPCR(std::vector <double> &aValues, const size_t &aRow,
              const size_t &aCol, const std::string &aPrecision);


#endif //MPCR_RBASICUTILITIES_HPP
