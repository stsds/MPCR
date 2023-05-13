
#include <data-units/DataType.hpp>
#include <adapters/RBasicUtilities.hpp>
#include <adapters/RBinaryOperations.hpp>
#include <adapters/RMathematicalOperations.hpp>
#include <adapters/RLinearAlgebra.hpp>




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

    /** Function that are not masked **/

    function("MPR.is.float", &RIsFloat,List::create(_["x"]));
    function("MPR.is.double", &RIsDouble,List::create(_["x"]));
    function("MPR.is.half", &RIsSFloat,List::create(_["x"]));
    function("MPR.rbind", &RRBind,List::create(_["x"],_["y"]));
    function("MPR.cbind", &RCBind,List::create(_["x"],_["y"]));
    function("MPR.is.na", &RIsNa, List::create(_[ "MPR" ], _[ "Index" ] = -1));
    function("MPR.Concatenate", &RConcatenate);
    function("MPR.ToNumericVector", &RToNumericVector);
    function("MPR.ToNumericMatrix", &RToNumericMatrix);
    function("MPR.ChangePrecision", &RChangePrecision);
    function("MPR.Add", &RPerformPlusDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MPR.Multiply", &RPerformMltDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MPR.Subtract", &RPerformMinusDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MPR.Divide", &RPerformDivDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MPR.Power", &RPerformPowDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));


    function("MPR.print", &RPrint,List::create(_["x"]));
    function("MPR.diag", &RGetDiagonal,List::create(_["x"]));
    function("MPR.min", &RGetMin,List::create(_["x"]));
    function("MPR.max", &RGetMax,List::create(_["x"]));
    function("MPR.na.omit", &RNaExclude,List::create(_["object"]));
    function("MPR.na.exclude", &RNaReplace,List::create(_["object"],_["value"]));
    function("MPR.nrow", &RGetNRow,List::create(_["x"]));
    function("MPR.ncol", &RGetNCol,List::create(_["x"]));
    function("MPR.object.size", &RObjectSize,List::create(_["x"]));
    function("MPR.str", &RPrint,List::create(_["object"]));
    function("MPR.show", &RGetType,List::create(_["object"]));
    function("MPR.rep", &RReplicate,
             List::create(_[ "x" ], _[ "count" ] = 0, _[ "len" ] = 0));
    function("MPR.sweep", &RSweep,List::create(_["x"],_["stat"],_["margin"],_["FUN"]));
    function("MPR.typeof", &RGetType,List::create(_["x"]));
    function("MPR.storage.mode", &RGetType,List::create(_["x"]));
    function("MPR.which.min", &RGetMinIdx,List::create(_["x"]));
    function("MPR.which.max", &RGetMaxIdx,List::create(_["x"]));
    function("MPR.scale", &RScaleDispatcher,List::create(_["x"],_["center"],_["scale"]));

    /** Math Functions **/

    function("MPR.abs", &RAbs,List::create(_["x"]));
    function("MPR.sqrt", &RSqrt,List::create(_["x"]));
    function("MPR.ceiling", &RCeiling,List::create(_["x"]));
    function("MPR.floor", &RFloor,List::create(_["x"]));
    function("MPR.trunc", &RTruncate,List::create(_["x"]));
    function("MPR.round", &RRound, List::create(_[ "x" ], _[ "digits" ] = 0));
    function("MPR.exp", &RExp,List::create(_["x"]));
    function("MPR.expm1", &RExp1m,List::create(_["x"]));
    function("MPR.gamma", &RGamma,List::create(_["x"]));
    function("MPR.lgamma", &RLGamma,List::create(_["x"]));
    function("MPR.is.finite", &RIsFinite,List::create(_["x"]));
    function("MPR.is.infinite", &RIsInFinite,List::create(_["x"]));
    function("MPR.is.nan", &RIsNan,List::create(_["x"]));
    function("MPR.log", &RLog, List::create(_[ "x" ], _[ "base" ] = 1));
    function("MPR.log10", &RLog10,List::create(_["x"]));
    function("MPR.log2", &RLog2,List::create(_["x"]));
    function("MPR.sin", &RSin,List::create(_["x"]));
    function("MPR.cos", &RCos,List::create(_["x"]));
    function("MPR.tan", &RTan,List::create(_["x"]));
    function("MPR.asin", &RASin,List::create(_["x"]));
    function("MPR.acos", &RACos,List::create(_["x"]));
    function("MPR.atan", &RATan,List::create(_["x"]));
    function("MPR.sinh", &RSinh,List::create(_["x"]));
    function("MPR.cosh", &RCosh,List::create(_["x"]));
    function("MPR.tanh", &RTanh,List::create(_["x"]));
    function("MPR.asinh", &RASinh,List::create(_["x"]));
    function("MPR.acosh", &RACosh,List::create(_["x"]));
    function("MPR.atanh", &RATanh,List::create(_["x"]));


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

    function("as.MPR", &RConvertToMPR,
             List::create(_[ "data" ], _[ "nrow" ] = 0, _[ "ncol" ] = 0,
                          _[ "precision" ]));


    /** Function to expose gemm , trsm , syrk **/
    function("MPR.gemm", &RGemm,
             List::create(_[ "a" ], _[ "b" ] = R_NilValue, _[ "c" ],
                          _[ "transpose_a" ] = false,
                          _[ "transpose_b" ] = false, _[ "alpha" ] = 1,
                          _[ "beta" ] = 0));

    function("MPR.trsm", &RTrsm,
             List::create(_[ "a" ], _[ "b" ], _[ "upper_triangle" ],
                          _[ "transpose" ] = false, _[ "side" ] = 'L',
                          _[ "alpha" ] = 1));

}
