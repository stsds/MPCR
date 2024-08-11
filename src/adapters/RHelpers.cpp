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


DataType *
RCopyMPR(DataType *aMatrix) {
    auto mat = new DataType(*aMatrix);
    return mat;
}