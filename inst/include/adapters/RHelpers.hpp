
#ifndef MPR_RHELPERS_HPP
#define MPR_RHELPERS_HPP


#include <data-units/DataType.hpp>


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


#endif //MPR_RHELPERS_HPP