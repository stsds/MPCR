
#ifndef MPR_RHELPERS_HPP
#define MPR_RHELPERS_HPP


//#include <data-units/DataType.hpp>
#include <data-units/MPRTile.hpp>


using namespace Rcpp;

/**
 * @brief
 * Change Vector of HALF Values to R-Logical Matrix
 *  1/TRUE  0/FALSE  INT_MIN=NA
 *
 * @param[in,out] aInput
 * MPR Object
 * @param[in] apDim
 * Dimensions to set R-Matrix With.
 *
 */
Rcpp::LogicalMatrix
ToLogicalMatrix(std::vector <int> &aInput, Dimensions *apDim);

/**
 * @brief
 * Change Vector of HALF Values to R-Logical Vector
 *  1/TRUE  0/FALSE  INT_MIN=NA
 *
 * @param[in,out] aInput
 * MPR Object
 *
 */
Rcpp::LogicalVector
ToLogicalVector(std::vector <int> &aInput);


void
RInsertTile(MPRTile *aMatrix, DataType *aTile, const size_t &aRowIdx,
           const size_t &aColIdx);

DataType *
RGetTile(MPRTile *aMatrix, const size_t &aRowIdx,const size_t &aColIdx);


#endif //MPR_RHELPERS_HPP