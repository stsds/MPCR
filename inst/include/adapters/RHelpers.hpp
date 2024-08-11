/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/
#ifndef MPCR_RHELPERS_HPP
#define MPCR_RHELPERS_HPP


#include <data-units/DataType.hpp>

using namespace Rcpp;

/**
 * @brief
 * Change Vector of HALF Values to R-Logical Matrix
 *  1/TRUE  0/FALSE  INT_MIN=NA
 *
 * @param[in,out] aInput
 * MPCR Object
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
 * MPCR Object
 *
 */
Rcpp::LogicalVector
ToLogicalVector(std::vector <int> &aInput);

/**
 * @brief
 * Creates a deepcopy of normal MPCR object
 *
 * @param[in] aMatrix
 * MPCR Object
 *
 * @returns
 *  a new Copy of MPCR Object
 */
DataType*
RCopyMPR(DataType *aMatrix);




#endif //MPCR_RHELPERS_HPP
