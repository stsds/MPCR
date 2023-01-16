
#include <data-units/DataType.hpp>
#include <adapters/RBasicUtilities.hpp>


/** Expose C++ class to R to be able to use Wrap and As
 *  Allows C++ to Send and Receive Class object from R
 **/
RCPP_EXPOSED_CLASS(DataType)

/** Expose C++ Object With the Given functions **/
RCPP_MODULE(DataTypeTest) {
    /** MPR Class **/
    using namespace Rcpp;
    class_ <DataType>("DataType")
        .constructor <size_t, std::string>()
        .property("IsMatrix", &DataType::IsMatrix)
        .property("Size", &DataType::GetSize)
        .property("Row", &DataType::GetNRow)
        .property("Col", &DataType::GetNCol)
        .method("PrintValues", &DataType::Print)
        .method("[[", &DataType::GetVal)
        .method("[", &DataType::GetValMatrix)
        .method("[[<-", &DataType::SetVal)
        .method("[<-", &DataType::SetValMatrix)
        .method("ToMatrix", &DataType::ToMatrix)
        .method("ToVector", &DataType::ToVector)
        .method("show", &RGetType);

    /** Basic Utilities **/

    function("cbind", &RCBind);
    function("rbind", &RRBind);
    function("diag", &RGetDiagonal);
    function("is.na", &RIsNa);
    function("is.float", &RIsFloat);
    function("is.double", &RIsDouble);
    function("is.sfloat", &RIsSFloat);
    function("min", &RGetMin);
    function("max", &RGetMax);
    function("na.omit", &RNaExclude);
    function("na.exclude", &RNaReplace);
    function("nrow", &RGetNRow);
    function("ncol", &RGetNCol);
    function("object.size", &RObjectSize);
    function("print", &RPrint);
    function("str", &RPrint);
    function("show", &RGetType);
    function("rep", &RReplicate);
    function("sweep", &RSweep);
    function("typeof", &RGetType);
    function("storage.mode", &RGetType);
    function("which.min", &RGetMinIdx);
    function("which.max", &RGetMaxIdx);
    function("getVal", &RGetElementVector);
    function("getVal", &RGetElementMatrix);
    function("RConcatenate", &RConcatenate);
    function("scale", &RScaleDispatcher);


}
