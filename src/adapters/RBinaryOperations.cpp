
#include <adapters/RHelpers.hpp>
#include <adapters/RBinaryOperations.hpp>
#include <utilities/MPRDispatcher.hpp>


using namespace mpr::precision;
using namespace mpr::operations::binary;


/************************** COMPARISONS ****************************/
SEXP
RGreaterThan(DataType *apInputA, DataType *apInputB) {
    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto precision_out = GetOutputPrecision(precision_a, precision_b);
    auto operation_comp = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);
    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;
    DISPATCHER(operation_comp, PerformCompareOperation, *apInputA, *apInputB,
               temp_out, ">", pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
RGreaterThan(DataType *apInputA, double aVal) {
    auto precision_a = apInputA->GetPrecision();
    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;

    SIMPLE_DISPATCH(precision_a, PerformCompareOperationSingle, *apInputA, aVal,
                    temp_out, ">", pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
RGreaterThanOrEqual(DataType *apInputA, DataType *apInputB) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto precision_out = GetOutputPrecision(precision_a, precision_b);
    auto operation_comp = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);
    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;
    DISPATCHER(operation_comp, PerformCompareOperation, *apInputA, *apInputB,
               temp_out, ">=", pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
RGreaterThanOrEqual(DataType *apInputA, double aVal) {

    auto precision_a = apInputA->GetPrecision();
    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;

    SIMPLE_DISPATCH(precision_a, PerformCompareOperationSingle, *apInputA, aVal,
                    temp_out, ">=", pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
RLessThan(DataType *apInputA, DataType *apInputB) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto precision_out = GetOutputPrecision(precision_a, precision_b);
    auto operation_comp = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);
    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;
    DISPATCHER(operation_comp, PerformCompareOperation, *apInputA, *apInputB,
               temp_out, "<", pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
RLessThan(DataType *apInputA, double aVal) {

    auto precision_a = apInputA->GetPrecision();
    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;

    SIMPLE_DISPATCH(precision_a, PerformCompareOperationSingle, *apInputA, aVal,
                    temp_out, "<", pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
RLessThanOrEqual(DataType *apInputA, DataType *apInputB) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto precision_out = GetOutputPrecision(precision_a, precision_b);
    auto operation_comp = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);
    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;
    DISPATCHER(operation_comp, PerformCompareOperation, *apInputA, *apInputB,
               temp_out, "<=", pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
RLessThanOrEqual(DataType *apInputA, double aVal) {

    auto precision_a = apInputA->GetPrecision();
    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;

    SIMPLE_DISPATCH(precision_a, PerformCompareOperationSingle, *apInputA, aVal,
                    temp_out, "<=", pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
REqual(DataType *apInputA, DataType *apInputB) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto precision_out = GetOutputPrecision(precision_a, precision_b);
    auto operation_comp = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);
    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;
    DISPATCHER(operation_comp, PerformEqualityOperation, *apInputA, *apInputB,
               temp_out, false, pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
REqual(DataType *apInputA, double aVal) {

    auto precision_a = apInputA->GetPrecision();

    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;
    SIMPLE_DISPATCH(precision_a, PerformEqualityOperationSingle, *apInputA,
                    aVal,
                    temp_out, false, pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
RNotEqual(DataType *apInputA, DataType *apInputB) {
    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto precision_out = GetOutputPrecision(precision_a, precision_b);
    auto operation_comp = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);
    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;
    DISPATCHER(operation_comp, PerformEqualityOperation, *apInputA, *apInputB,
               temp_out, true, pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


SEXP
RNotEqual(DataType *apInputA, double aVal) {
    auto precision_a = apInputA->GetPrecision();

    std::vector <int> temp_out;
    Dimensions *pDim = nullptr;
    SIMPLE_DISPATCH(precision_a, PerformEqualityOperationSingle, *apInputA,
                    aVal,
                    temp_out, true, pDim)

    if (pDim != nullptr) {
        auto matrix = ToLogicalMatrix(temp_out, pDim);
        delete pDim;
        return matrix;
    }
    auto vec = ToLogicalVector(temp_out);
    return vec;
}


/************************** OPERATIONS ****************************/

DataType *
RPerformPlus(DataType *apInputA, DataType *apInputB) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto pOutput = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);
    DISPATCHER(operation_comb, PerformOperation, *apInputA, *apInputB, *pOutput,
               "+")
    return pOutput;
}


DataType *
RPerformPlus(DataType *apInputA, double aVal, std::string aPrecision) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = precision_a;
    if (aPrecision != "") {
        precision_b = GetInputPrecision(aPrecision);
    }
    auto precision_out = GetOutputPrecision(precision_a, precision_b);

    auto pOutput = new DataType(precision_out);

    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);

    DISPATCHER(operation_comb, PerformOperationSingle, *apInputA, aVal,
               *pOutput,
               "+")

    return pOutput;
}


DataType *
RPerformMinus(DataType *apInputA, DataType *apInputB) {
    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto pOutput = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);
    DISPATCHER(operation_comb, PerformOperation, *apInputA, *apInputB, *pOutput,
               "-")
    return pOutput;
}


DataType *
RPerformMinus(DataType *apInputA, double aVal, std::string aPrecision) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = precision_a;
    if (aPrecision != "") {
        precision_b = GetInputPrecision(aPrecision);
    }
    auto precision_out = GetOutputPrecision(precision_a, precision_b);

    auto pOutput = new DataType(precision_out);

    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);

    DISPATCHER(operation_comb, PerformOperationSingle, *apInputA, aVal,
               *pOutput,
               "-")

    return pOutput;
}


DataType *
RPerformMult(DataType *apInputA, DataType *apInputB) {
    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto pOutput = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);
    DISPATCHER(operation_comb, PerformOperation, *apInputA, *apInputB, *pOutput,
               "*")
    return pOutput;
}


DataType *
RPerformMult(DataType *apInputA, double aVal, std::string aPrecision) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = precision_a;
    if (aPrecision != "") {
        precision_b = GetInputPrecision(aPrecision);
    }
    auto precision_out = GetOutputPrecision(precision_a, precision_b);

    auto pOutput = new DataType(precision_out);

    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);

    DISPATCHER(operation_comb, PerformOperationSingle, *apInputA, aVal,
               *pOutput,
               "*")

    return pOutput;
}


DataType *
RPerformDiv(DataType *apInputA, DataType *apInputB) {
    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto pOutput = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);
    DISPATCHER(operation_comb, PerformOperation, *apInputA, *apInputB, *pOutput,
               "/")
    return pOutput;
}


DataType *
RPerformDiv(DataType *apInputA, double aVal, std::string aPrecision) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = precision_a;
    if (aPrecision != "") {
        precision_b = GetInputPrecision(aPrecision);
    }
    auto precision_out = GetOutputPrecision(precision_a, precision_b);

    auto pOutput = new DataType(precision_out);

    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);

    DISPATCHER(operation_comb, PerformOperationSingle, *apInputA, aVal,
               *pOutput,
               "/")

    return pOutput;
}


DataType *
RPerformPow(DataType *apInputA, DataType *apInputB) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto pOutput = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);
    DISPATCHER(operation_comb, PerformOperation, *apInputA, *apInputB, *pOutput,
               "^")
    return pOutput;

}


DataType *
RPerformPow(DataType *apInputA, double aVal, std::string aPrecision) {

    auto precision_a = apInputA->GetPrecision();
    auto precision_b = precision_a;
    if (aPrecision != "") {
        precision_b = GetInputPrecision(aPrecision);
    }
    auto precision_out = GetOutputPrecision(precision_a, precision_b);

    auto pOutput = new DataType(precision_out);

    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                precision_out);

    DISPATCHER(operation_comb, PerformOperationSingle, *apInputA, aVal,
               *pOutput,
               "^")

    return pOutput;
}


/************************** DISPATCHERS ****************************/

DataType *
RPerformPlusDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision) {

    if (TYPEOF(aObj) == REALSXP || TYPEOF(aObj) == INTSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformPlus(apInputA, val, aPrecision);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MMPR Object",
                -1);
        }
        return RPerformPlus(apInputA, temp_mpr);
    }
}


DataType *
RPerformMinusDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision) {
    if (TYPEOF(aObj) == REALSXP || TYPEOF(aObj) == INTSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformMinus(apInputA, val, aPrecision);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MMPR Object",
                -1);
        }
        return RPerformMinus(apInputA, temp_mpr);
    }
}


DataType *
RPerformMltDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision) {
    if (TYPEOF(aObj) == REALSXP || TYPEOF(aObj) == INTSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformMult(apInputA, val, aPrecision);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MMPR Object",
                -1);
        }
        return RPerformMult(apInputA, temp_mpr);
    }
}


DataType *
RPerformDivDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision) {
    if (TYPEOF(aObj) == REALSXP || TYPEOF(aObj) == INTSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformDiv(apInputA, val, aPrecision);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MMPR Object",
                -1);
        }
        return RPerformDiv(apInputA, temp_mpr);
    }
}


DataType *
RPerformPowDispatcher(DataType *apInputA, SEXP aObj, std::string aPrecision) {
    if (TYPEOF(aObj) == REALSXP || TYPEOF(aObj) == INTSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformPow(apInputA, val, aPrecision);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MMPR Object",
                -1);
        }
        return RPerformPow(apInputA, temp_mpr);
    }

}


/************************** CONVERTERS ****************************/

std::vector <double>
RToNumericVector(DataType *apInputA) {
    auto vec = apInputA->ConvertToNumericVector();
    return *vec;
}


SEXP
RToNumericMatrix(DataType *apInputA) {
    auto matrix = apInputA->ConvertToRMatrix();
    return *matrix;
}


void
RChangePrecision(DataType *apInputA, std::string aPrecision) {
    auto precision = GetInputPrecision(aPrecision);
    apInputA->ConvertPrecision(precision);
}


