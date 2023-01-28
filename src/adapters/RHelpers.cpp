
#include <adapters/RHelpers.hpp>


Rcpp::LogicalMatrix&
ToLogicalMatrix(std::vector <int> &aInput, Dimensions *apDim) {
    auto matrix = new Rcpp::LogicalMatrix(apDim->GetNRow(),
                                          apDim->GetNCol());
    auto itr_vec = aInput.begin();
    for (auto i = 0; i < apDim->GetNRow(); i++) {
        auto row = matrix->row(i);
        for (auto itr = row.begin(); itr != row.end(); itr++) {
            *itr = *itr_vec;
            itr_vec++;
        }
    }

    return *matrix;
}

Rcpp::LogicalVector&
ToLogicalVector(std::vector<int> &aInput){
    auto vec = new Rcpp::LogicalVector(aInput.size());
    vec->assign(aInput.begin(), aInput.end());
    return *vec;
}