/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <operations/BinaryOperations.hpp>
#include <operations/helpers/BasicOperationsHelper.hpp>
#include <operations/helpers/BinaryOperationsHelper.hpp>


using namespace mpcr::operations;
using namespace std;


template <typename T, typename X, typename Y>
void
binary::PerformOperation(DataType &aInputA, DataType &aInputB,
                         DataType &aOutput,
                         const string &aFun) {

    auto size_a = aInputA.GetSize();
    auto size_b = aInputB.GetSize();
    auto size_out = std::max(size_a, size_b);
    binary::CheckDimensions(aInputA, aInputB);

    aOutput.ClearUp();
    aOutput.SetSize(size_out);

    auto pInput_data_a = (T *) aInputA.GetData();
    auto pInput_data_b = (X *) aInputB.GetData();
    auto pOutput_data = new Y[size_out];

    if (aInputA.IsMatrix()) {
        aOutput.SetDimensions(aInputA.GetNRow(), aInputA.GetNCol());

    } else if (aInputB.IsMatrix()) {
        aOutput.SetDimensions(aInputB.GetNRow(), aInputB.GetNCol());
    }


    RUN_BINARY_OP(pInput_data_a, pInput_data_b, pOutput_data, aFun, size_a,
                  size_b, size_out)

    aOutput.SetData((char *) pOutput_data);

}


template <typename T, typename X, typename Y>
void
binary::PerformOperationSingle(DataType &aInputA, const double &aVal,
                               DataType &aOutput, const string &aFun) {


    aOutput.ClearUp();
    auto size = aInputA.GetSize();

    if (aInputA.IsMatrix()) {
        aOutput.ToMatrix(aInputA.GetNRow(), aInputA.GetNCol());
    } else {
        aOutput.SetSize(size);
    }

    auto pData_input = (T *) aInputA.GetData();
    auto pData_out = new Y[size];

    RUN_BINARY_OP_SINGLE(pData_input, aVal, pData_out, aFun, size)

    aOutput.SetData((char *) pData_out);
}


template <typename T, typename X, typename Y>
void
binary::PerformCompareOperation(DataType &aInputA, DataType &aInputB,
                                vector <int> &aOutput, const string &aFun,
                                Dimensions *&apDimensions) {

    binary::CheckDimensions(aInputA, aInputB);
    auto size_in_a = aInputA.GetSize();
    auto size_in_b = aInputB.GetSize();
    auto size_out = std::max(size_in_a, size_in_b);

    auto pData_in_a = (T *) aInputA.GetData();
    auto pData_in_b = (X *) aInputB.GetData();


    aOutput.clear();
    aOutput.resize(size_out);

    if (!apDimensions) {
        delete apDimensions;
        apDimensions = new Dimensions();
    }
    bool is_matrix = false;

    if (aInputA.IsMatrix()) {
        apDimensions->SetNRow(aInputA.GetNRow());
        apDimensions->SetNCol(aInputA.GetNCol());
        is_matrix = true;
    } else if (aInputB.IsMatrix()) {
        apDimensions->SetNRow(aInputB.GetNRow());
        apDimensions->SetNCol(aInputB.GetNCol());
        is_matrix = true;
    }

    size_t idx = 0;

    RUN_COMPARE_OP_SIMPLE(pData_in_a, pData_in_b, aOutput, aFun, size_in_b,
                          size_in_a, size_out)

    if (!is_matrix) {
        delete apDimensions;
        apDimensions = nullptr;
    }
}


template <typename T>
void
binary::PerformCompareOperationSingle(DataType &aInputA, const double &aVal,
                                      std::vector <int> &aOutput,
                                      const std::string &aFun,
                                      Dimensions *&apDimensions) {
    if (aInputA.IsMatrix()) {
        delete apDimensions;
        apDimensions = new Dimensions();
        apDimensions->SetNRow(aInputA.GetNRow());
        apDimensions->SetNCol(aInputA.GetNCol());
    }

    auto size_in_a = aInputA.GetSize();
    auto pData_in_a = (T *) aInputA.GetData();

    aOutput.clear();
    aOutput.resize(size_in_a);

    RUN_COMPARE_OP_SINGLE(pData_in_a, aVal, aOutput, aFun, size_in_a)

}


void
binary::CheckDimensions(DataType &aInputA, DataType &aInputB) {

    auto size_a = aInputA.GetSize();
    auto size_b = aInputB.GetSize();
    auto both_is_matrix = ( aInputA.IsMatrix() && aInputB.IsMatrix());

    if (aInputA.IsMatrix() || aInputB.IsMatrix()) {
        if (size_a != size_b) {
            MPCR_API_EXCEPTION("Matrix dims do not match the length of object",
                              -1);
        } else if (both_is_matrix) {
            if (aInputA.GetNRow() != aInputB.GetNRow()) {
                MPCR_API_EXCEPTION(
                    "Matrix dims do not match the length of object, non-conformable arrays ",
                    -1);
            }
        }
    }

    auto a_is_bigger = ( size_a > size_b );

    if (a_is_bigger && size_b % size_a != 0 ||
        !a_is_bigger && size_a % size_b != 0) {
        MPCR_API_WARN(
            "longer object length is not a multiple of shorter object length",
            -1);
    }

}


template <typename T, typename X, typename Y>
void
binary::PerformEqualityOperation(DataType &aInputA, DataType &aInputB,
                                 std::vector <int> &aOutput,
                                 const bool &aIsNotEqual,
                                 Dimensions *&apDimensions) {
    binary::CheckDimensions(aInputA, aInputB);
    auto size_in_a = aInputA.GetSize();
    auto size_in_b = aInputB.GetSize();
    auto size_out = std::max(size_in_a, size_in_b);

    auto pData_in_a = (T *) aInputA.GetData();
    auto pData_in_b = (X *) aInputB.GetData();


    aOutput.clear();
    aOutput.resize(size_out);

    if (!apDimensions) {
        delete apDimensions;
        apDimensions = new Dimensions();
    }
    bool is_matrix = false;

    if (aInputA.IsMatrix()) {
        apDimensions->SetNRow(aInputA.GetNRow());
        apDimensions->SetNCol(aInputA.GetNCol());
        is_matrix = true;
    } else if (aInputB.IsMatrix()) {
        apDimensions->SetNRow(aInputB.GetNRow());
        apDimensions->SetNCol(aInputB.GetNCol());
        is_matrix = true;
    }


    auto epsilon = std::numeric_limits <Y>::epsilon();

    for (auto i = 0; i < size_out; i++) {
        auto element_a = pData_in_a[ i % size_in_a ];
        auto element_b = pData_in_b[ i % size_in_b ];
        if (isnan(element_a) || isnan(element_b)) {
            aOutput[ i ] = INT_MIN;
        } else {
            auto error = fabs((Y) ( element_a - element_b ));
            if (error < epsilon) {
                aOutput[ i ] = !aIsNotEqual;
            } else {
                aOutput[ i ] = aIsNotEqual;
            }
        }
    }


    if (!is_matrix) {
        delete apDimensions;
        apDimensions = nullptr;
    }
}


template <typename T>
void
binary::PerformEqualityOperationSingle(DataType &aInputA, double &aVal,
                                       std::vector <int> &aOutput,
                                       const bool &aIsNotEqual,
                                       Dimensions *&apDimensions) {


    if (aInputA.IsMatrix()) {
        delete apDimensions;
        apDimensions = new Dimensions();
        apDimensions->SetNRow(aInputA.GetNRow());
        apDimensions->SetNCol(aInputA.GetNCol());
    }

    auto size_in_a = aInputA.GetSize();
    auto pData_in_a = (T *) aInputA.GetData();

    aOutput.clear();
    aOutput.resize(size_in_a);
    auto epsilon = std::numeric_limits <T>::epsilon();
    if (isnan(aVal)) {
        aOutput.assign(size_in_a, INT_MIN);
        return;
    }


    for (auto i = 0; i < size_in_a; i++) {
        auto element_a = pData_in_a[ i ];
        if (isnan(element_a)) {
            aOutput[ i ] = INT_MIN;
        } else {
            auto error = fabs((T) ( element_a - aVal ));
            if (error < epsilon) {
                aOutput[ i ] = !aIsNotEqual;
            } else {
                aOutput[ i ] = aIsNotEqual;
            }
        }
    }

}


INSTANTIATE(void, binary::PerformEqualityOperation, DataType &aInputA,
            DataType &aInputB, std::vector <int> &aOutput,
            const bool &aIsNotEqual, Dimensions *&apDimensions)

INSTANTIATE(void, binary::PerformOperationSingle, DataType &aInputA,
            const double &aVal, DataType &aOutput, const string &aFun)

INSTANTIATE(void, binary::PerformOperation, DataType &aInputA,
            DataType &aInputB, DataType &aOutput, const string &aFun)

INSTANTIATE(void, binary::PerformCompareOperation, DataType &aInputA,
            DataType &aInputB, vector <int> &aOutput, const string &aFun,
            Dimensions *&aDimensions)

SIMPLE_INSTANTIATE(void, binary::PerformCompareOperationSingle, DataType &
    aInputA, const double &aVal, std::vector <int> &aOutput,
                   const std::string &aFun, Dimensions *&apDimensions)

SIMPLE_INSTANTIATE(void, binary::PerformEqualityOperationSingle,
                   DataType &aInputA, double &aVal, std::vector <int> &
                       aOutput, const bool &aIsNotEqual,
                   Dimensions *&apDimensions)
