/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <adapters/RMathematicalOperations.hpp>
#include <adapters/RHelpers.hpp>
#include <utilities/MPCRDispatcher.hpp>


using namespace mpcr::precision;
using namespace mpcr::operations;


DataType *
RAbs(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformRoundOperation, *aInput, *pOutput,
                    "abs")
    return pOutput;
}


DataType *
RSqrt(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::SquareRoot, *aInput, *pOutput)
    return pOutput;
}


DataType *
RCeiling(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformRoundOperation, *aInput, *pOutput,
                    "ceil")
    return pOutput;
}


DataType *
RFloor(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformRoundOperation, *aInput, *pOutput,
                    "floor")
    return pOutput;
}


DataType *
RTruncate(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformRoundOperation, *aInput, *pOutput,
                    "trunc")
    return pOutput;
}


DataType *
RRound(DataType *aInput, const int &aDecimalPlaces) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::Round, *aInput, *pOutput, aDecimalPlaces)
    return pOutput;
}


DataType *
RExp(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::Exponential, *aInput, *pOutput)
    return pOutput;
}


DataType *
RExp1m(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::Exponential, *aInput, *pOutput, true)
    return pOutput;
}


DataType *
RGamma(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::Gamma, *aInput, *pOutput)
    return pOutput;
}


DataType *
RLGamma(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::Gamma, *aInput, *pOutput, true)
    return pOutput;
}


SEXP
RIsFinite(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    std::vector <int> output;
    SIMPLE_DISPATCH(precision, math::IsFinite, *aInput, output)
    if (aInput->IsMatrix()) {
        Dimensions temp(aInput->GetNRow(), aInput->GetNCol());
        return ToLogicalMatrix(output, &temp);
    } else {
        return ToLogicalVector(output);
    }
}


SEXP
RIsInFinite(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    std::vector <int> output;
    SIMPLE_DISPATCH(precision, math::IsInFinite, *aInput, output)
    if (aInput->IsMatrix()) {
        Dimensions temp(aInput->GetNRow(), aInput->GetNCol());
        return ToLogicalMatrix(output, &temp);
    } else {
        return ToLogicalVector(output);
    }
}


SEXP
RIsNan(DataType *aInput) {
    std::vector <int> output;
    Dimensions *pDim = nullptr;
    aInput->IsNA(pDim);
    if (aInput->IsMatrix()) {
        return ToLogicalMatrix(output, pDim);
    } else {
        return ToLogicalVector(output);
    }
}


DataType *
RLog(DataType *aInput, int aBase) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::Log, *aInput, *pOutput, aBase)
    return pOutput;
}


DataType *
RLog10(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::Log, *aInput, *pOutput, 10)
    return pOutput;
}


DataType *
RLog2(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::Log, *aInput, *pOutput, 2)
    return pOutput;
}


DataType *
RSin(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformTrigOperation, *aInput, *pOutput,
                    "sin")
    return pOutput;
}


DataType *
RCos(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformTrigOperation, *aInput, *pOutput,
                    "cos")
    return pOutput;
}


DataType *
RTan(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformTrigOperation, *aInput, *pOutput,
                    "tan")
    return pOutput;
}


DataType *
RASin(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformInverseTrigOperation, *aInput,
                    *pOutput, "asin")
    return pOutput;
}


DataType *
RACos(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformInverseTrigOperation, *aInput,
                    *pOutput, "acos")
    return pOutput;
}


DataType *
RATan(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformInverseTrigOperation, *aInput,
                    *pOutput, "atan")
    return pOutput;
}


DataType *
RSinh(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformTrigOperation, *aInput, *pOutput,
                    "sinh")
    return pOutput;
}


DataType *
RCosh(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformTrigOperation, *aInput, *pOutput,
                    "cosh")
    return pOutput;
}


DataType *
RTanh(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformTrigOperation, *aInput, *pOutput,
                    "tanh")
    return pOutput;
}


DataType *
RASinh(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformInverseTrigOperation, *aInput,
                    *pOutput, "asinh")
    return pOutput;
}


DataType *
RACosh(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformInverseTrigOperation, *aInput,
                    *pOutput, "acosh")
    return pOutput;
}


DataType *
RATanh(DataType *aInput) {
    auto precision = aInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, math::PerformInverseTrigOperation, *aInput,
                    *pOutput, "atanh")
    return pOutput;
}

