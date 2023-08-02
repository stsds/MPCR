
#include <data-units/DataType.hpp>
#include <adapters/RBasicUtilities.hpp>
#include <adapters/RBinaryOperations.hpp>
#include <adapters/RMathematicalOperations.hpp>
#include <adapters/RLinearAlgebra.hpp>
#include <adapters/RHelpers.hpp>




/** Expose C++ class to R to be able to use Wrap and As
 *  Allows C++ to Send and Receive Class object from R
 **/
RCPP_EXPOSED_CLASS(DataType)

/** Expose C++ Object With the Given functions **/
RCPP_MODULE(MMPR) {


    /** MPR Class **/
    using namespace Rcpp;


    /** Basic Utilities **/
    class_ <DataType>("MMPR")
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
        .method("MMPR.GetVal", &DataType::GetVal)
        .method("MMPR.GetValMatrix", &DataType::GetValMatrix)
        .method("MMPR.SetVal", &DataType::SetVal)
        .method("MMPR.SetValMatrix", &DataType::SetValMatrix)
        .method("NotEqual", &DataType::NotEqualDispatcher);

    /** Function that are not masked **/

    function("MMPR.is.single", &RIsFloat,List::create(_["x"]));
    function("MMPR.is.float", &RIsFloat,List::create(_["x"]));
    function("MMPR.is.double", &RIsDouble,List::create(_["x"]));
    function("MMPR.is.half", &RIsSFloat,List::create(_["x"]));
    function("MMPR.rbind", &RRBind,List::create(_["x"],_["y"]));
    function("MMPR.cbind", &RCBind,List::create(_["x"],_["y"]));
    function("MMPR.is.na", &RIsNa, List::create(_[ "object" ], _[ "index" ] = -1));
    function("MMPR.na.exclude", &RNaReplace,List::create(_["object"],_["value"]));
    function("MMPR.na.omit", &RNaExclude,List::create(_["object"]));
    function("MMPR.object.size", &RObjectSize,List::create(_["x"]));
    function("MMPR.Concatenate", &RConcatenate,List::create(_["x"]));
    function("MMPR.ToNumericVector", &RToNumericVector,List::create(_["x"]));
    function("MMPR.ToNumericMatrix", &RToNumericMatrix,List::create(_["x"]));
    function("MMPR.ChangePrecision", &RChangePrecision,List::create(_["x"],_["precision"]));
    function("MMPR.Add", &RPerformPlusDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MMPR.Multiply", &RPerformMltDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MMPR.Subtract", &RPerformMinusDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MMPR.Divide", &RPerformDivDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MMPR.Power", &RPerformPowDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));


    function("MMPR.print", &RPrint,List::create(_["x"]));
    function("MMPR.diag", &RGetDiagonal,List::create(_["x"]));
    function("MMPR.min", &RGetMin,List::create(_["x"]));
    function("MMPR.max", &RGetMax,List::create(_["x"]));
    function("MMPR.nrow", &RGetNRow,List::create(_["x"]));
    function("MMPR.ncol", &RGetNCol,List::create(_["x"]));
    function("MMPR.str", &RPrint,List::create(_["object"]));
    function("MMPR.show", &RGetType,List::create(_["object"]));
    function("MMPR.rep", &RReplicate,
             List::create(_[ "x" ], _[ "count" ] = 0, _[ "len" ] = 0));
    function("MMPR.sweep", &RSweep,List::create(_["x"],_["stat"],_["margin"],_["FUN"]));
    function("MMPR.typeof", &RGetType,List::create(_["x"]));
    function("MMPR.storage.mode", &RGetType,List::create(_["x"]));
    function("MMPR.which.min", &RGetMinIdx,List::create(_["x"]));
    function("MMPR.which.max", &RGetMaxIdx,List::create(_["x"]));
    function("MMPR.scale", &RScaleDispatcher,List::create(_["x"],_["center"],_["scale"]));

    /** Math Functions **/

    function("MMPR.abs", &RAbs,List::create(_["x"]));
    function("MMPR.sqrt", &RSqrt,List::create(_["x"]));
    function("MMPR.ceiling", &RCeiling,List::create(_["x"]));
    function("MMPR.floor", &RFloor,List::create(_["x"]));
    function("MMPR.trunc", &RTruncate,List::create(_["x"]));
    function("MMPR.round", &RRound, List::create(_[ "x" ], _[ "digits" ] = 0));
    function("MMPR.exp", &RExp,List::create(_["x"]));
    function("MMPR.expm1", &RExp1m,List::create(_["x"]));
    function("MMPR.gamma", &RGamma,List::create(_["x"]));
    function("MMPR.lgamma", &RLGamma,List::create(_["x"]));
    function("MMPR.is.finite", &RIsFinite,List::create(_["x"]));
    function("MMPR.is.infinite", &RIsInFinite,List::create(_["x"]));
    function("MMPR.is.nan", &RIsNan,List::create(_["x"]));
    function("MMPR.log", &RLog, List::create(_[ "x" ], _[ "base" ] = 1));
    function("MMPR.log10", &RLog10,List::create(_["x"]));
    function("MMPR.log2", &RLog2,List::create(_["x"]));
    function("MMPR.sin", &RSin,List::create(_["x"]));
    function("MMPR.cos", &RCos,List::create(_["x"]));
    function("MMPR.tan", &RTan,List::create(_["x"]));
    function("MMPR.asin", &RASin,List::create(_["x"]));
    function("MMPR.acos", &RACos,List::create(_["x"]));
    function("MMPR.atan", &RATan,List::create(_["x"]));
    function("MMPR.sinh", &RSinh,List::create(_["x"]));
    function("MMPR.cosh", &RCosh,List::create(_["x"]));
    function("MMPR.tanh", &RTanh,List::create(_["x"]));
    function("MMPR.asinh", &RASinh,List::create(_["x"]));
    function("MMPR.acosh", &RACosh,List::create(_["x"]));
    function("MMPR.atanh", &RATanh,List::create(_["x"]));


    /** Linear Algebra **/

    function("MMPR.backsolve", &RBackSolve,
             List::create(_[ "r" ], _[ "x" ], _[ "k" ] = -1,
                          _[ "upper.tri" ] = true, _[ "transpose" ] = false));
    function("MMPR.forwardsolve", &RBackSolve,
             List::create(_[ "r" ], _[ "x" ], _[ "k" ] = -1,
                          _[ "upper.tri" ] = false, _[ "transpose" ] = false));
    function("MMPR.chol", &RCholesky);
    function("MMPR.chol2inv", &RCholeskyInv);
    function("MMPR.crossprod", &RCrossProduct,
             List::create(_[ "x" ], _[ "y" ] = R_NilValue));
    function("MMPR.tcrossprod", &RTCrossProduct,
             List::create(_[ "x" ], _[ "y" ] = R_NilValue));
    function("MMPR.eigen", &REigen,
             List::create(_[ "x" ], _[ "only.values" ] = false));
    function("MMPR.isSymmetric", &RIsSymmetric);
    function("MMPR.svd", &RSVD,
             List::create(_[ "x" ], _[ "nu" ] = -1, _[ "nv" ] = -1,
                          _[ "Transpose" ] = true));
    function("MMPR.La.svd", &RSVD,
             List::create(_[ "x" ], _[ "nu" ] = -1, _[ "nv" ] = -1,
                          _[ "Transpose" ] = false));
    function("MMPR.norm", &RNorm, List::create(_[ "x" ], _[ "type" ] = "O"));
    function("MMPR.qr", &RQRDecomposition,List::create(_["x"],_["tol"]= 1e-07));
    function("MMPR.qr.Q", &RQRDecompositionQ,
             List::create(_[ "qr" ], _[ "qraux" ], _[ "complete" ] = false,
                          _[ "Dvec" ] = R_NilValue));
    function("MMPR.qr.R", &RQRDecompositionR);
    function("MMPR.rcond", &RRCond,
             List::create(_[ "x" ], _[ "norm" ] = "O", _[ "useInv" ] = false));
    function("MMPR.solve", &RSolve);
    function("MMPR.t", &RTranspose);
    function("MMPR.qr.qy", &RQRDecompositionQy);
    function("MMPR.qr.qty", &RQRDecompositionQty);

    function("as.MMPR", &RConvertToMPR,
             List::create(_[ "data" ], _[ "nrow" ] = 0, _[ "ncol" ] = 0,
                          _[ "precision" ]));


    /** Function to expose gemm , trsm , syrk **/
    function("MMPR.gemm", &RGemm,
             List::create(_[ "a" ], _[ "b" ] = R_NilValue, _[ "c" ],
                          _[ "transpose_a" ] = false,
                          _[ "transpose_b" ] = false, _[ "alpha" ] = 1,
                          _[ "beta" ] = 0));

    function("MMPR.trsm", &RTrsm,
             List::create(_[ "a" ], _[ "b" ], _[ "upper_triangle" ],
                          _[ "transpose" ] = false, _[ "side" ] = 'L',
                          _[ "alpha" ] = 1));


    function("MMPR.copy",&RCopyMPR,List::create(_["x"]));


}
