
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