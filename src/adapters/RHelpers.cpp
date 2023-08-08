/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MMPR is an R package provided by the STSDS group at KAUST
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
RInsertTile(MPRTile *aMatrix, DataType *aTile, const size_t &aRowIdx,
            const size_t &aColIdx) {
    auto new_obj = new DataType(*aTile);
    aMatrix->InsertTile(new_obj, aRowIdx - 1, aColIdx - 1);
}


DataType *
RGetTile(MPRTile *aMatrix, const size_t &aRowIdx, const size_t &aColIdx) {
    auto pOutput = aMatrix->GetTile(aRowIdx - 1, aColIdx - 1);
    auto new_obj = new DataType(*pOutput);
    return new_obj;
}


MPRTile *
RCopyMPRTile(MPRTile *aMatrix) {
    auto tile_matrix = new MPRTile(*aMatrix);
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
RGetSerializeTile(MPRTile *aMatrix, const size_t &aRowIdx,
                 const size_t &aColIdx) {
    auto pOutput = aMatrix->GetTile(aRowIdx - 1, aColIdx - 1);
    return pOutput->RSerialize();
}