
#include <data-units/DataType.hpp>
#include <adapters/RBasicUtilities.hpp>
#include <adapters/RBinaryOperations.hpp>


/** Expose C++ class to R to be able to use Wrap and As
 *  Allows C++ to Send and Receive Class object from R
 **/
RCPP_EXPOSED_CLASS(DataType)



// [[Rcpp::export]]
DataType *
Plus(DataType &aInputA, DataType &aInputB) {

    Rcpp::Rcout<<"hereeeeeeeeeeeee"<<std::endl;
    //    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RPerformPlus(this, val, "");
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
////        }
//        DataType temp=aInput;
//        return RPerformPlus(this, aInput );
//    }
    return &aInputA;
}


/** Expose C++ Object With the Given functions **/
RCPP_MODULE(MPR) {
    /** MPR Class **/
    using namespace Rcpp;
//
//    DataType *
//    (*xAdd)(DataType &, DataType &) =&operator+;

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
        .method("=", &DataType::operator =)
//        .method(">", &DataType::operator >)
//        .method(">=", &DataType::operator >=)
//        .method("<", &DataType::operator <)
//        .method("<=", &DataType::operator <=)
//        .method("==", &DataType::operator ==)
//        .method("!=", &DataType::operator !=)
//        .method("operator+",&DataType::operator+)
//        .method("-", &DataType::operator -)
//        .method("*", &DataType::operator *)
//        .method("/", &DataType::operator /)
//        .method("^", &DataType::operator ^)
        ;

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
    function("Plus", &RPerformPlusDispatcher);
//    function("`+`",&RPerformPlusDispatcher);
//    function("ToNumericVector", &RToNumericVector);
//    function("ToNumericMatrix", &RToNumericMatrix);
//    function("ChangePrecision", &RChangePercision);
//    function("Add", &RPerformPlusDispatcher,
//             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
//    function("Multiply", &RPerformMltDispatcher,
//             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
//    function("Subtract", &RPerformMinusDispatcher,
//             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
//    function("Divide", &RPerformDivDispatcher,
//             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
//    function("Power", &RPerformPowDispatcher,
//             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));


}
