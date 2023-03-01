
#include <data-units/DataType.hpp>
#include <adapters/RBasicUtilities.hpp>
#include <adapters/RBinaryOperations.hpp>
#include <adapters/RMathematicalOperations.hpp>
#include <adapters/RLinearAlgebra.hpp>




/** Expose C++ class to R to be able to use Wrap and As
 *  Allows C++ to Send and Receive Class object from R
 **/
//RCPP_EXPOSED_CLASS(DataType)
RCPP_EXPOSED_AS(DataType)

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
        .method("MPR.GetVal", &DataType::GetVal)
        .method("MPR.GetValMatrix", &DataType::GetValMatrix)
        .method("MPR.SetVal", &DataType::SetVal)
        .method("MPR.SetValMatrix", &DataType::SetValMatrix)
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
    function("MPR.print", &RPrint);
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


    /** Math Functions **/

    function("abs", &RAbs);
    function("sqrt", &RSqrt);
    function("ceiling", &RCeiling);
    function("floor", &RFloor);
    function("trunc", &RTruncate);
    function("round", &RRound, List::create(_[ "x" ], _[ "digits" ] = 0));
    function("exp", &RExp);
    function("exp1m", &RExp1m);
    function("gamma", &RGamma);
    function("lgamma", &RLGamma);
    function("is.finite", &RIsFinite);
    function("is.infinite", &RIsInFinite);
    function("is.nan", &RIsNan);
    function("log", &RLog, List::create(_[ "x" ], _[ "base" ] = 1));
    function("log10", &RLog10);
    function("log2", &RLog2);
    function("sin", &RSin);
    function("cos", &RCos);
    function("tan", &RTan);
    function("asin", &RASin);
    function("acos", &RACos);
    function("atan", &RATan);
    function("sinh", &RSinh);
    function("cosh", &RCosh);
    function("tanh", &RTanh);
    function("asinh", &RASinh);
    function("acosh", &RACosh);
    function("atanh", &RATanh);

    /** Linear Algebra **/

    function("MPR.backsolve", &RBackSolve,
             List::create(_[ "r" ], _[ "x" ], _[ "k" ] = -1,
                          _[ "upper.tri" ] = true, _[ "transpose" ] = false));
    function("MPR.forwardsolve", &RBackSolve,
             List::create(_[ "r" ], _[ "x" ], _[ "k" ] = -1,
                          _[ "upper.tri" ] = false, _[ "transpose" ] = false));
    function("MPR.chol", &RCholesky);
    function("MPR.chol2inv", &RCholeskyInv);
    function("MPR.crossprod", &RCrossProduct,
             List::create(_[ "x" ], _[ "y" ] = R_NilValue));
    function("MPR.tcrossprod", &RTCrossProduct,
             List::create(_[ "x" ], _[ "y" ] = R_NilValue));
    function("MPR.eigen", &REigen,
             List::create(_[ "x" ], _[ "only.values" ] = false));
    function("MPR.isSymmetric", &RIsSymmetric);
    function("MPR.svd", &RSVD,
             List::create(_[ "x" ], _[ "nu" ] = -1, _[ "nv" ] = -1,
                          _[ "Transpose" ] = true));
    function("MPR.La.svd", &RSVD,
             List::create(_[ "x" ], _[ "nu" ] = -1, _[ "nv" ] = -1,
                          _[ "Transpose" ] = false));
    function("MPR.norm", &RNorm, List::create(_[ "x" ], _[ "type" ] = "O"));
    function("MPR.qr", &RQRDecomposition);
    function("MPR.qr.Q", &RQRDecompositionQ,
             List::create(_[ "qr" ], _[ "qraux" ], _[ "complete" ] = false,
                          _[ "Dvec" ] = R_NilValue));
    function("MPR.qr.R", &RQRDecompositionR);
    function("MPR.rcond", &RRCond,
             List::create(_[ "x" ], _[ "norm" ] = "O", _[ "useInv" ] = false));
    function("MPR.solve", &RSolve);
    function("MPR.t", &RTranspose);
    function("MPR.qr.qy", &RQRDecompositionQy);
    function("MPR.qr.qty", &RQRDecompositionQty);


}
