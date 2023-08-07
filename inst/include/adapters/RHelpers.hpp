/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MMPR is an R package provided by the STSDS group at KAUST
 *
 **/
#ifndef MPR_RHELPERS_HPP
#define MPR_RHELPERS_HPP


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

/**
 * @brief
 * Updates a Tile in MPRTile Object, by copying the tile and inserting the new
 * tile to avoid R ownership problem.
 *
 * @param[in] aMatrix
 * MPRTile Matrix
 * @param[in] aTile
 * Tile containing new data that needs to be updated.
 * @param[in] aRowIdx
 * Tile Row idx
 * @param[in] aColIdx
 * Tile Col idx
 *
 */
void
RInsertTile(MPRTile *aMatrix, DataType *aTile, const size_t &aRowIdx,
           const size_t &aColIdx);
/**
 * @brief
 * Get a Tile from MPRTile Object
 *
 * @param[in] aMatrix
 * MPRTile Matrix
 * @param[in] aRowIdx
 * Tile Row idx
 * @param[in] aColIdx
 * Tile Col idx
 *
 *
 * @returns
 *  a copy of the tile at the idx [aRowIdx,aColIdx] to avoid ownership problem
 *  in R
 */
DataType *
RGetTile(MPRTile *aMatrix, const size_t &aRowIdx,const size_t &aColIdx);

/**
 * @brief
 * Creates a deepcopy of MPRTile Matrix
 *
 * @param[in] aMatrix
 * MPRTile Matrix
 *
 * @returns
 *  a new Copy of MPRTile Matrix
 */
MPRTile *
RCopyMPRTile(MPRTile *aMatrix);

/**
 * @brief
 * Creates a deepcopy of normal MPR object
 *
 * @param[in] aMatrix
 * MPR Object
 *
 * @returns
 *  a new Copy of MPR Object
 */
DataType*
RCopyMPR(DataType *aMatrix);

#endif //MPR_RHELPERS_HPP