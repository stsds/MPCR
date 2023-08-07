/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MMPR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPR_RBASICUTILITIES_HPP
#define MPR_RBASICUTILITIES_HPP

#include <data-units/DataType.hpp>


/**
 * @brief
 * R Adapter for Combining two Matrices by columns
 *
 * @param[in] apInputA
 * MPR Matrix one
 * @param[in] apInputB
 * MPR Matrix two
 *
 * @return
 * MPR Matrix holding combined data
 *
 */
DataType *
RCBind(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R Adapter for Combining two Matrices by rows
 *
 * @param[in] apInputA
 * MPR Matrix one
 * @param[in] apInputB
 * MPR Matrix two
 *
 * @return
 * MPR Matrix holding combined data
 *
 */
DataType *
RRBind(DataType *apInputA, DataType *apInputB);

/**
 * @brief
 * R Adapter for Checking if MPR object is 16-bit Precision
 *
 * @param[in] apInput
 * MPR Object
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
 * R Adapter for Checking if MPR object is 32-bit Precision
 *
 * @param[in] apInput
 * MPR Object
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
 * R Adapter for Checking if MPR object is 64-bit Precision
 *
 * @param[in] apInput
 * MPR Object
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
 * MPR object to replicate
 * @param[in] aSize
 * Size of output vector
 * @param[in] aLength
 * Length of Output Value
 *
 * @returns
 * MPR Vector Holding Replicated Data
 *
 */
DataType *
RReplicate(DataType *apInput, size_t aSize, size_t aLength);

/**
 * @brief
 * R Adapter for Removing NA values from Vector.
 *
 * @param[in,out] apInput
 * MPR Object.
 *
 */
void
RNaExclude(DataType *apInput);

/**
 * @brief
 * R Adapter for Replacing NA values with a given value.
 *
 * @param[in,out] apInput
 * MPR Object.
 * @param[in] aValue
 * Value to use it instead of NA's
 *
 */
void
RNaReplace(DataType *apInput, double aValue);

/**
 * @brief
 * R adapter for Getting Diagonal of a Matrix.
 * MPR Object must be a Matrix
 *
 * @param[in] apInput
 * MPR Matrix
 *
 * @returns
 * MPR Object holding Diagonals
 *
 */
DataType *
RGetDiagonal(DataType *apInput);

/**
 * @brief
 * R adapter for Getting Diagonal of a Matrix.
 * MPR can be a Vector , but dims must be passed
 *
 * @param[in] apInput
 * MPR object can be Vector or Matrix
 * @param[out] aRow
 * Number of Rows
 * @param[in] aCol
 * Number of Cols
 *
 * @returns
 * MPR Object holding Diagonals
 *
 */
DataType *
RGetDiagonalWithDims(DataType *apInput, size_t aRow, size_t aCol);

/**
 * @brief
 * R Adapter for Printing string indicating whether it's 16/32/64 Bit Precision
 *
 * @param[in] apInput
 * MPR object can be Vector or Matrix
 *
 */
void
RGetType(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Min Element in Array
 *
 * @param[in] apInput
 * MPR object can be Vector or Matrix
 *
 * @returns
 * MPR Object holding Minimum Val (Same Precision)
 *
 */
DataType *
RGetMin(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Min Element Index in Array
 *
 * @param[in] apInput
 * MPR object can be Vector or Matrix
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
 * MPR object can be Vector or Matrix
 *
 * @returns
 * MPR Object holding Maximum Val (Same Precision)
 *
 */
DataType *
RGetMax(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Max Element Index in Array
 *
 * @param[in] apInput
 * MPR object can be Vector or Matrix
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
 * MPR Matrix
 * @param[in] apStats
 * the value(s) that should be used in the operation
 * @param[in] aMargin
 * aMargin = 1 means row; aMargin = otherwise means column.
 * @param[in] aOperation
 * char containing operation (+,-,*,/)
 *
 * @returns
 * MPR Vector holding Data After Applying Sweep
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
 * MPR Object
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
 * Get total size of Memory used by MPR Object
 *
 * @param[in] apInput
 * MPR Object
 *
 * @returns
 * Total size of Memory used by MPR Object
 *
 */
size_t
RObjectSize(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Number of Rows
 *
 * @param[in] apInput
 * MPR Object
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
 * MPR Object
 *
 * @returns
 * Number of Rows in a Matrix
 */

size_t
RGetNCol(DataType *apInput);

/**
 * @brief
 * R Adapter for Printing Information about MPR object
 * Dimensions-Matrix/Vector-Values ,and Precisions.
 *
 * @param[in] apInput
 * MPR Object.
 *
 */
void
RPrint(DataType *apInput);

/**
 * @brief
 * R Adapter for Getting Element with Idx from MPR Vector as MPR Object
 *
 * @param[in] apInput
 * MPR Object
 * @param[in] aIndex
 * Index of Data
 *
 * @returns
 * MPR Object holding element at idx
 *
 */
DataType *
RGetElementVector(DataType *apInput, size_t aIndex);

/**
 * @brief
 * R Adapter for Getting Element with Idx [row][col] from MPR Matrix
 * as MPR Object
 *
 * @param[in] apInput
 * MPR Object
 * @param[in] aRow
 * Row Idx
 * @param[in] aCol
 * Col Idx
 *
 * @returns
 * MPR Object holding element at idx
 *
 */
DataType *
RGetElementMatrix(DataType *apInput, size_t aRowIdx,
                  size_t aColIdx);

/**
 * @brief
 * R Adapter for Concatenating List of MPR Vectors into one MPR Vector.
 * This Function Casts the SEXP pointer to DataTypes pointers , And Check a Magic
 * Number inside the MPR Class to determine if its a MPR object or Not.
 *
 * Warning:
 * There's a very Small Possibility that the Passed Objects' Magic Number is
 * the Same as DataType , in this case , The behavior of the function is unexpected.
 * So the User should check whether all Objects are MPR Objects or not.
 *
 * @param[in] aList
 * List Of SEXP
 *
 * @returns
 * MPR Vector containing all values in all lists (Precision = Highest Precision
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
 * MPR Object.
 * @param[in] apCenter
 * numeric-alike MPR vector of length equal to the number of
 * columns of aInput.
 * @param[out] apScale
 * numeric-alike MPR vector of length equal to the number of
 * columns of aInput.
 * @returns
 * MPR Object with the same size and shape after centering and/or scaling.
 *
 */
DataType *
RScale(DataType *apInput, DataType *apCenter, DataType *apScale);

/**
 * @brief
 *  R Adapter for Centering and/or Scaling the columns of a numeric matrix.
 *
 * @param[in] apInputA
 * MPR Object.
 * @param[in] aCenter
 * bool if true centering is done using column mean ,else no centering is done.
 * @param[out] apScale
 * numeric-alike MPR vector of length equal to the number of
 * columns of aInput.
 * @returns
 * MPR Object with the same size and shape after centering and/or scaling.
 *
 */
DataType *
RScale(DataType *apInput, bool aCenter, DataType *apScale);

/**
 * @brief
 *  R Adapter for Centering and/or Scaling the columns of a numeric matrix.
 *
 * @param[in] apInputA
 * MPR Object.
 * @param[in] apCenter
 * numeric-alike MPR vector of length equal to the number of
 * columns of aInput.
 * @param[out] aScale
 * bool if true scaling is done using column standard deviation,else no scaling
 * is done.
 * @returns
 * MPR Object with the same size and shape after centering and/or scaling.
 *
 */
DataType *
RScale(DataType *apInput, DataType *apCenter, bool aScale);

/**
 * @brief
 *  R Adapter for Centering and/or Scaling the columns of a numeric matrix.
 *
 * @param[in] apInputA
 * MPR Object.
 * @param[in] aCenter
 * bool if true centering is done using column mean ,else no centering is done.
 * @param[out] aScale
 * bool if true scaling is done using column standard deviation,else no scaling
 * is done.
 * @returns
 * MPR Object with the same size and shape after centering and/or scaling.
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
 * MPR Object.
 * @returns
 * MPR Object with the same size and shape after centering and/or scaling.
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
 * MPR Object with the same size and shape after centering and/or scaling.
 *
 */
DataType *
RScaleDispatcher(SEXP a, SEXP b, SEXP c);

/**
 * @brief
 *  Converts R vector or Matrix to MPR object.
 *  if aRow or aCol = zero , MPR vector will be created , else MPR Matrix.
 *
 * @param[in] aValues
 * R vector/Matrix holding values to create MPR object from.
 * @param[in] aRow
 * Number of Rows in case of creating an MPR Matrix .
 * @param[in] aCol
 * Number of Cols in case of creating an MPR Matrix .
 * @param[in] aPrecision
 * Required Precision of the created MPR Object.
 *
 * @returns
 * New MPR Object constructed from the given inputs
 *
 */
DataType *
RConvertToMPR(std::vector <double> &aValues, const size_t &aRow,
              const size_t &aCol, const std::string &aPrecision);


#endif //MPR_RBASICUTILITIES_HPP
