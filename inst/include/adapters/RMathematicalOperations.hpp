

#ifndef MPR_RMATHEMATICALOPERATIONS_HPP
#define MPR_RMATHEMATICALOPERATIONS_HPP

#include <operations/MathematicalOperations.hpp>


DataType *
RAbs(DataType *aInput);

DataType *
RSqrt(DataType *aInput);

DataType *
RCeiling(DataType *aInput);

DataType *
RFloor(DataType *aInput);

DataType *
RTruncate(DataType *aInput);

DataType *
RRound(DataType *aInput, const int &aDecimalPlaces);

DataType *
RExp(DataType *aInput);

DataType *
RExp1m(DataType *aInput);

DataType *
RGamma(DataType *aInput);

DataType *
RLGamma(DataType *aInput);

SEXP
RIsFinite(DataType *aInput);

SEXP
RIsInFinite(DataType *aInput);

SEXP
RIsNan(DataType *aInput);

DataType *
RLog(DataType *aInput);


DataType *
RLog10(DataType *aInput);


DataType *
RLog2(DataType *aInput);


DataType *
RSin(DataType *aInput);


DataType *
RCos(DataType *aInput);


DataType *
RTan(DataType *aInput);


DataType *
RASin(DataType *aInput);


DataType *
RACos(DataType *aInput);


DataType *
RATan(DataType *aInput);

DataType *
RSinh(DataType *aInput);


DataType *
RCosh(DataType *aInput);


DataType *
RTanh(DataType *aInput);


DataType *
RASinh(DataType *aInput);


DataType *
RACosh(DataType *aInput);


DataType *
RATanh(DataType *aInput);


#endif //MPR_RMATHEMATICALOPERATIONS_HPP
