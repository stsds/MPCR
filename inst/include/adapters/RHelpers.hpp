/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/
#ifndef MPCR_RHELPERS_HPP
#define MPCR_RHELPERS_HPP


#include <data-units/MPCRTile.hpp>


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
 * Updates a Tile in MPCRTile Object, by copying the tile and inserting the new
 * tile to avoid R ownership problem.
 *
 * @param[in] aMatrix
 * MPCRTile Matrix
 * @param[in] aTile
 * Tile containing new data that needs to be updated.
 * @param[in] aRowIdx
 * Tile Row idx
 * @param[in] aColIdx
 * Tile Col idx
 *
 */
void
RInsertTile(MPCRTile *aMatrix, DataType *aTile, const size_t &aRowIdx,
           const size_t &aColIdx);
/**
 * @brief
 * Get a Tile from MPCRTile Object
 *
 * @param[in] aMatrix
 * MPCRTile Matrix
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
RGetTile(MPCRTile *aMatrix, const size_t &aRowIdx,const size_t &aColIdx);

/**
 * @brief
 * Creates a deepcopy of MPCRTile Matrix
 *
 * @param[in] aMatrix
 * MPCRTile Matrix
 *
 * @returns
 *  a new Copy of MPCRTile Matrix
 */
MPCRTile *
RCopyMPCRTile(MPCRTile *aMatrix);

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

/**
 * @brief
 * R version to Serialize DataType object as a Raw Vector
 *
 * @returns
 * vector of bytes containing DataType object as a stream of bytes
 *
 */
Rcpp::RawVector
SerializeTile(DataType *aInput);

/**
 * @brief
 * R version to DeSerialize Stream of bytes to MPCR Object
 *
 * @param[in] aInput
 * vector of bytes containing DataType object as a stream of bytes
 *
 */
DataType *
DeSerializeTile(Rcpp::RawVector aInput);

/**
 * @brief
 * R version to Serialize Tile object as a Raw Vector from Tile-Matrix without
 * copying
 *
 * @returns
 * vector of bytes containing DataType object as a stream of bytes
 *
 */
Rcpp::RawVector
RGetSerializeTile(MPCRTile *aMatrix, const size_t &aRowIdx,const size_t &aColIdx);



#endif //MPCR_RHELPERS_HPP