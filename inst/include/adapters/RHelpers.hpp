
#ifndef MPR_RHELPERS_HPP
#define MPR_RHELPERS_HPP


#include <data-units/DataType.hpp>

using namespace Rcpp;

Rcpp::LogicalMatrix&
ToLogicalMatrix(std::vector<int> &aInput,Dimensions *apDim);

Rcpp::LogicalVector&
ToLogicalVector(std::vector<int> &aInput);


#endif //MPR_RHELPERS_HPP
