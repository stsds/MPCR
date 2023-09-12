/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <adapters/RHelpers.hpp>


Rcpp::LogicalMatrix
ToLogicalMatrix(std::vector <int> &aInput, Dimensions *apDim) {
    auto matrix = Rcpp::LogicalMatrix(apDim->GetNRow(),
                                      apDim->GetNCol(), aInput.data());

    return matrix;
}


Rcpp::LogicalVector
ToLogicalVector(std::vector <int> &aInput) {
    auto vec = Rcpp::LogicalVector(aInput.size());
    vec.assign(aInput.begin(), aInput.end());
    return vec;
}


void
RInsertTile(MPCRTile *aMatrix, DataType *aTile, const size_t &aRowIdx,
            const size_t &aColIdx) {
    auto new_obj = new DataType(*aTile);
    aMatrix->InsertTile(new_obj, aRowIdx - 1, aColIdx - 1);
}


DataType *
RGetTile(MPCRTile *aMatrix, const size_t &aRowIdx, const size_t &aColIdx) {
    auto pOutput = aMatrix->GetTile(aRowIdx - 1, aColIdx - 1);
    auto new_obj = new DataType(*pOutput);
    return new_obj;
}


MPCRTile *
RCopyMPCRTile(MPCRTile *aMatrix) {
    auto tile_matrix = new MPCRTile(*aMatrix);
    return tile_matrix;
}


DataType *
RCopyMPR(DataType *aMatrix) {
    auto mat = new DataType(*aMatrix);
    return mat;
}


RawVector
SerializeTile(DataType *aInput) {
    return aInput->RSerialize();
}


DataType *
DeSerializeTile(Rcpp::RawVector aInput) {
    return DataType::RDeSerialize(aInput);
}


Rcpp::RawVector
RGetSerializeTile(MPCRTile *aMatrix, const size_t &aRowIdx,
                 const size_t &aColIdx) {
    auto pOutput = aMatrix->GetTile(aRowIdx - 1, aColIdx - 1);
    return pOutput->RSerialize();
}