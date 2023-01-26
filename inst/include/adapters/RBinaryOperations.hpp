

#ifndef MPR_RBINARYOPERATIONS_HPP
#define MPR_RBINARYOPERATIONS_HPP

#include <operations/BinaryOperations.hpp>


/************************** COMPARISONS ****************************/
SEXP
RGreaterThan(DataType *apInputA, DataType *apInputB);


SEXP
RGreaterThan(DataType *apInputA, double aVal);


SEXP
RGreaterThanOrEqual(DataType *apInputA, DataType *apInputB);


SEXP
RGreaterThanOrEqual(DataType *apInputA, double aVal);


SEXP
RLessThan(DataType *apInputA, DataType *apInputB);


SEXP
RLessThan(DataType *apInputA, double aVal);


SEXP
RLessThanOrEqual(DataType *apInputA, DataType *apInputB);


SEXP
RLessThanOrEqual(DataType *apInputA, double aVal);


SEXP
REqual(DataType *apInputA, DataType *apInputB);


SEXP
REqual(DataType *apInputA, double aVal);


SEXP
RNotEqual(DataType *apInputA, DataType *apInputB);


SEXP
RNotEqual(DataType *apInputA, double aVal);


/************************** OPERATIONS ****************************/
DataType *
RPerformPlus(DataType *apInputA, DataType *apInputB);

DataType *
RPerformPlus(DataType *apInputA, double aVal, std::string aPrecision);

DataType *
RPerformMinus(DataType *apInputA, DataType *apInputB);

DataType *
RPerformMinus(DataType *apInputA, double aVal, std::string aPrecision);

DataType *
RPerformMult(DataType *apInputA, DataType *apInputB);

DataType *
RPerformMult(DataType *apInputA, double aVal, std::string aPrecision);

DataType *
RPerformDiv(DataType *apInputA, DataType *apInputB);

DataType *
RPerformDiv(DataType *apInputA, double aVal, std::string aPrecision);

DataType *
RPerformPow(DataType *apInputA, DataType *apInputB);

DataType *
RPerformPow(DataType *apInputA, double aVal, std::string aPrecision);

/************************** DISPATCHERS ****************************/

DataType *
RPerformPlusDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

DataType *
RPerformMinusDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

DataType *
RPerformMltDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

DataType *
RPerformDivDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

DataType *
RPerformPowDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision);

SEXP
RGreaterThanDispatcher(DataType *apInputA, SEXP aObj);

SEXP
RGreaterThanOrEqualDispatcher(DataType *apInputA, SEXP aObj);

SEXP
RLessThanDispatcher(DataType *apInputA, SEXP aObj);


SEXP
RLessThanOrEqualDispatcher(DataType *apInputA, SEXP aObj);


SEXP
REqualDispatcher(DataType *apInputA, SEXP aObj);

SEXP
RNotEqualDispatcher(DataType *apInputA, SEXP aObj);

/************************** CONVERTERS ****************************/

std::vector <double>
RToNumericVector(DataType *apInputA);

SEXP
RToNumericMatrix(DataType *apInputA);

void
RChangePrecision(DataType *apInputA, std::string aPrecision);


#endif //MPR_RBINARYOPERATIONS_HPP
