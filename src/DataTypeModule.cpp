
#include <data-units/DataType.hpp>
#include <adapters/RBasicUtilities.hpp>
#include <adapters/RBinaryOperations.hpp>


/** Expose C++ class to R to be able to use Wrap and As
 *  Allows C++ to Send and Receive Class object from R
 **/
RCPP_EXPOSED_CLASS(DataType)

/** Expose C++ Object With the Given functions **/
RCPP_MODULE(MPR) {

    /** MPR Class **/
    using namespace Rcpp;

    /** Basic Utilities **/
    class_ <DataType>("MPR")
        .constructor <size_t, std::string>()
        .property("IsMatrix", &DataType::IsMatrix)
        .property("Size", &DataType::GetSize)
        .property("Row", &DataType::GetNRow)
        .property("Col", &DataType::GetNCol)
        .method("PrintValues", &DataType::Print)
        .method("[[", &DataType::GetVal)
        .method("[[<-", &DataType::SetVal)
        .method("ToMatrix", &DataType::ToMatrix)
        .method("ToVector", &DataType::ToVector)
        .method("show", &RGetType)
        .method("PerformPlus", &DataType::PerformPlusDispatcher)
        .method("PerformMinus", &DataType::PerformMinusDispatcher)
        .method("PerformMult", &DataType::PerformMultDispatcher)
        .method("PerformDiv", &DataType::PerformDivDispatcher)
        .method("PerformPow", &DataType::PerformPowDispatcher)
        .method("GreaterThan", &DataType::GreaterThanDispatcher)
        .method("GreaterEqual", &DataType::GreaterThanOrEqualDispatcher)
        .method("LessThan", &DataType::LessThanDispatcher)
        .method("LessEqual", &DataType::LessThanOrEqualDispatcher)
        .method("EqualEqual", &DataType::EqualDispatcher)
        .method("NotEqual", &DataType::NotEqualDispatcher);

    function("cbind", &RCBind);
    function("rbind", &RRBind);
    function("diag", &RGetDiagonal);
    function("is.na", &RIsNa, List::create(_[ "MPR" ], _[ "Index" ] = -1));
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
    function("rep", &RReplicate,
             List::create(_[ "MPR" ], _[ "count" ] = 0, _[ "len" ] = 0));
    function("sweep", &RSweep);
    function("typeof", &RGetType);
    function("storage.mode", &RGetType);
    function("which.min", &RGetMinIdx);
    function("which.max", &RGetMaxIdx);
    function("getVal", &RGetElementVector);
    function("getVal", &RGetElementMatrix);
    function("RConcatenate", &RConcatenate);
    function("scale", &RScaleDispatcher);
    function("ToNumericVector", &RToNumericVector);
    function("ToNumericMatrix", &RToNumericMatrix);
    function("ChangePrecision", &RChangePrecision);
    function("Add", &RPerformPlusDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("Multiply", &RPerformMltDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("Subtract", &RPerformMinusDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("Divide", &RPerformDivDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("Power", &RPerformPowDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));


}
