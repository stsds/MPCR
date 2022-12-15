
#include <data-units/DataType.hpp>
#include <Rcpp.h>


DataType::DataType(size_t aSize, float aVal) {
    DataTypeInit(aSize, aVal);
}


void DataType::DataTypeInit(size_t aSize, float aVal) {
    this->mVecSize = aSize;
    this->mpData = new float[aSize];
    for (auto i = 0; i < this->mVecSize; i++) {
        this->mpData[i] = aVal;
    }
}


void DataType::PrintVal() {
    for (auto i = 0; i < this->mVecSize; i++) {
        Rcpp::Rcout << this->mpData[i] << std::endl;
    }
}


float *
DataType::GetData() {
    return this->mpData;
}


size_t
DataType::GetSize() {
    return this->mVecSize;
}


