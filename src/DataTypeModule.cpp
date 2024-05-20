/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <adapters/RBasicUtilities.hpp>
#include <adapters/RBinaryOperations.hpp>
#include <adapters/RMathematicalOperations.hpp>
#include <adapters/RLinearAlgebra.hpp>
#include <adapters/RHelpers.hpp>
#include <adapters/RContextManager.hpp>




/** Expose C++ class to R to be able to use Wrap and As
 *  Allows C++ to Send and Receive Class object from R
 **/
RCPP_EXPOSED_CLASS(DataType)

/** Expose C++ Object With the Given functions **/
RCPP_MODULE(MPCR) {


    /** MPCR Class **/
    using namespace Rcpp;


    /** Basic Utilities **/
    class_ <DataType>("MPCR")
        .constructor <size_t, std::string,std::string>()
        .property("IsMatrix", &DataType::IsMatrix)
        .property("Size", &DataType::GetSize)
        .property("Row", &DataType::GetNRow)
        .property("Col", &DataType::GetNCol)
        .method("PrintValues", &DataType::Print)
        .method("ToMatrix", &DataType::ToMatrix)
        .method("ToVector", &DataType::ToVector)
        .method("Sum", &DataType::Sum)
        .method("SquareSum", &DataType::SquareSum)
        .method("Prod", &DataType::Product)
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
        .method("MPCR.GetVal", &DataType::GetVal)
        .method("MPCR.GetValMatrix", &DataType::GetValMatrix)
        .method("MPCR.SetVal", &DataType::SetVal)
        .method("MPCR.SetValMatrix", &DataType::SetValMatrix)
        .method("NotEqual", &DataType::NotEqualDispatcher);

    /** Function that are not masked **/

    function("MPCR.is.single", &RIsFloat,List::create(_["x"]));
    function("MPCR.is.float", &RIsFloat,List::create(_["x"]));
    function("MPCR.is.double", &RIsDouble,List::create(_["x"]));
    function("MPCR.is.half", &RIsSFloat,List::create(_["x"]));
    function("MPCR.rbind", &RRBind,List::create(_["x"],_["y"]));
    function("MPCR.cbind", &RCBind,List::create(_["x"],_["y"]));
    function("MPCR.is.na", &RIsNa, List::create(_[ "object" ], _[ "index" ] = -1));
    function("MPCR.na.exclude", &RNaReplace,List::create(_["object"],_["value"]));
    function("MPCR.na.omit", &RNaExclude,List::create(_["object"]));
    function("MPCR.object.size", &RObjectSize,List::create(_["x"]));
    function("MPCR.Concatenate", &RConcatenate,List::create(_["x"]));
    function("MPCR.ToNumericVector", &RToNumericVector,List::create(_["x"]));
    function("MPCR.ToNumericMatrix", &RToNumericMatrix,List::create(_["x"]));
    function("MPCR.ChangePrecision", &RChangePrecision,List::create(_["x"],_["precision"]));
    function("MPCR.Add", &RPerformPlusDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MPCR.Multiply", &RPerformMltDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MPCR.Subtract", &RPerformMinusDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MPCR.Divide", &RPerformDivDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));
    function("MPCR.Power", &RPerformPowDispatcher,
             List::create(_[ "x" ], _[ "y" ], _[ "Precision" ] = ""));


    function("MPCR.print", &RPrint,List::create(_["x"]));
    function("MPCR.diag", &RGetDiagonal,List::create(_["x"]));
    function("MPCR.min", &RGetMin,List::create(_["x"]));
    function("MPCR.max", &RGetMax,List::create(_["x"]));
    function("MPCR.nrow", &RGetNRow,List::create(_["x"]));
    function("MPCR.ncol", &RGetNCol,List::create(_["x"]));
    function("MPCR.str", &RPrint,List::create(_["object"]));
    function("MPCR.show", &RGetType,List::create(_["object"]));
    function("MPCR.rep", &RReplicate,
             List::create(_[ "x" ], _[ "count" ] = 0, _[ "len" ] = 0));
    function("MPCR.sweep", &RSweep,List::create(_["x"],_["stat"],_["margin"],_["FUN"]));
    function("MPCR.typeof", &RGetType,List::create(_["x"]));
    function("MPCR.storage.mode", &RGetType,List::create(_["x"]));
    function("MPCR.which.min", &RGetMinIdx,List::create(_["x"]));
    function("MPCR.which.max", &RGetMaxIdx,List::create(_["x"]));
    function("MPCR.scale", &RScaleDispatcher,List::create(_["x"],_["center"],_["scale"]));

    /** Math Functions **/

    function("MPCR.abs", &RAbs,List::create(_["x"]));
    function("MPCR.sqrt", &RSqrt,List::create(_["x"]));
    function("MPCR.ceiling", &RCeiling,List::create(_["x"]));
    function("MPCR.floor", &RFloor,List::create(_["x"]));
    function("MPCR.trunc", &RTruncate,List::create(_["x"]));
    function("MPCR.round", &RRound, List::create(_[ "x" ], _[ "digits" ] = 0));
    function("MPCR.exp", &RExp,List::create(_["x"]));
    function("MPCR.expm1", &RExp1m,List::create(_["x"]));
    function("MPCR.gamma", &RGamma,List::create(_["x"]));
    function("MPCR.lgamma", &RLGamma,List::create(_["x"]));
    function("MPCR.is.finite", &RIsFinite,List::create(_["x"]));
    function("MPCR.is.infinite", &RIsInFinite,List::create(_["x"]));
    function("MPCR.is.nan", &RIsNan,List::create(_["x"]));
    function("MPCR.log", &RLog, List::create(_[ "x" ], _[ "base" ] = 1));
    function("MPCR.log10", &RLog10,List::create(_["x"]));
    function("MPCR.log2", &RLog2,List::create(_["x"]));
    function("MPCR.sin", &RSin,List::create(_["x"]));
    function("MPCR.cos", &RCos,List::create(_["x"]));
    function("MPCR.tan", &RTan,List::create(_["x"]));
    function("MPCR.asin", &RASin,List::create(_["x"]));
    function("MPCR.acos", &RACos,List::create(_["x"]));
    function("MPCR.atan", &RATan,List::create(_["x"]));
    function("MPCR.sinh", &RSinh,List::create(_["x"]));
    function("MPCR.cosh", &RCosh,List::create(_["x"]));
    function("MPCR.tanh", &RTanh,List::create(_["x"]));
    function("MPCR.asinh", &RASinh,List::create(_["x"]));
    function("MPCR.acosh", &RACosh,List::create(_["x"]));
    function("MPCR.atanh", &RATanh,List::create(_["x"]));


    /** Linear Algebra **/

    function("MPCR.backsolve", &RBackSolve,
             List::create(_[ "r" ], _[ "x" ], _[ "k" ] = -1,
                          _[ "upper.tri" ] = true, _[ "transpose" ] = false));
    function("MPCR.forwardsolve", &RBackSolve,
             List::create(_[ "r" ], _[ "x" ], _[ "k" ] = -1,
                          _[ "upper.tri" ] = false, _[ "transpose" ] = false));
    function("MPCR.chol", &RCholesky);
    function("MPCR.chol2inv", &RCholeskyInv);
    function("MPCR.crossprod", &RCrossProduct,
             List::create(_[ "x" ], _[ "y" ] = R_NilValue));
    function("MPCR.tcrossprod", &RTCrossProduct,
             List::create(_[ "x" ], _[ "y" ] = R_NilValue));
    function("MPCR.eigen", &REigen,
             List::create(_[ "x" ], _[ "only.values" ] = false));
    function("MPCR.isSymmetric", &RIsSymmetric);
    function("MPCR.svd", &RSVD,
             List::create(_[ "x" ], _[ "nu" ] = -1, _[ "nv" ] = -1,
                          _[ "Transpose" ] = true));
    function("MPCR.La.svd", &RSVD,
             List::create(_[ "x" ], _[ "nu" ] = -1, _[ "nv" ] = -1,
                          _[ "Transpose" ] = false));
    function("MPCR.norm", &RNorm, List::create(_[ "x" ], _[ "type" ] = "O"));
    function("MPCR.qr", &RQRDecomposition,List::create(_["x"]));
    function("MPCR.qr.Q", &RQRDecompositionQ,
             List::create(_[ "qr" ], _[ "qraux" ], _[ "complete" ] = false,
                          _[ "Dvec" ] = R_NilValue));
    function("MPCR.qr.R", &RQRDecompositionR);
    function("MPCR.rcond", &RRCond,
             List::create(_[ "x" ], _[ "norm" ] = "O", _[ "useInv" ] = false));
    function("MPCR.solve", &RSolve,
             List::create(_[ "a" ], _[ "b" ] = R_NilValue,
                          _[ "internal_precision" ] = "same"));
    function("MPCR.t", &RTranspose);
    function("MPCR.qr.qy", &RQRDecompositionQy);
    function("MPCR.qr.qty", &RQRDecompositionQty);

    function("as.MPCR", &RConvertToMPCR,
             List::create(_[ "data" ], _[ "nrow" ] = 0, _[ "ncol" ] = 0,
                          _[ "precision" ],_["placement"]="CPU"));


    /** Function to expose gemm , trsm , syrk **/
    function("MPCR.gemm", &RGemm,
             List::create(_[ "a" ], _[ "b" ] = R_NilValue, _[ "c" ],
                          _[ "transpose_a" ] = false,
                          _[ "transpose_b" ] = false, _[ "alpha" ] = 1,
                          _[ "beta" ] = 0));

    function("MPCR.trsm", &RTrsm,
             List::create(_[ "a" ], _[ "b" ], _[ "upper_triangle" ],
                          _[ "transpose" ] = false, _[ "side" ] = 'L',
                          _[ "alpha" ] = 1));


    function("MPCR.copy",&RCopyMPR,List::create(_["x"]));


    /** Run Context Functions **/

    function("MPCR.SetOperationPLacement",&SetOperationPlacement);
    function("MPCR.GetOperationPLacement",&GetOperationPlacement);

}
