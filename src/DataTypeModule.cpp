
#include <data-units/DataType.hpp>

/** Expose C++ class to R to be able to use Wrap and As
 *  Allows C++ to Send and Receive Class object from R
 **/
RCPP_EXPOSED_CLASS(DataType)

/** Expose C++ Object With the Given functions **/
RCPP_MODULE(DataTypeTest){
    using namespace Rcpp;
    class_<DataType>("DataType")
        .constructor<size_t,std::string>()
        .property("IsMatrix",&DataType::IsMatrix)
        .property("Size",&DataType::GetSize)
        .property("Row",&DataType::GetNRow)
        .property("Col",&DataType::GetNCol)
        .method("PrintValues",&DataType::Print)
        .method("[[",&DataType::GetVal)
        .method("[[<-",&DataType::SetVal)
        .method("ToMatrix",&DataType::ToMatrix)
        .method("ToVector",&DataType::ToVector)
        ;
}
