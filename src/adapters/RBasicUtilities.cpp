
#include <operations/BasicOperations.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <adapters/RBasicUtilities.hpp>


using namespace mpr::operations;
using namespace mpr::precision;


/**
 * This File Contains R adapters for C++ functions since R sends and receives
 * pointers to objects. and to assure proper dispatching.
 **/

DataType *
RCBind(DataType *apInputA, DataType *apInputB) {
    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto output = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);

    DISPATCHER(operation_comb, basic::ColumnBind, *apInputA, *apInputB, *output)
    return output;
}


DataType *
RRBind(DataType *apInputA, DataType *apInputB) {
    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto output = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);

    DISPATCHER(operation_comb, basic::RowBind, *apInputA, *apInputB, *output)
    return output;
}


bool
RIsSFloat(DataType *apInput) {
    return basic::IsSFloat(*apInput);
}


bool
RIsFloat(DataType *apInput) {
    return basic::IsFloat(*apInput);
}


bool
RIsDouble(DataType *apInput) {
    return basic::IsDouble(*apInput);
}


DataType *
RReplicate(DataType *apInput, size_t aSize) {
    auto precision = apInput->GetPrecision();
    auto output = new DataType(precision);
    SIMPLE_DISPATCH(precision, basic::Replicate, *apInput, *output, aSize)
    return output;
}


void
RNaExclude(DataType *apInput) {
    SIMPLE_DISPATCH(apInput->GetPrecision(), basic::NAExclude, *apInput)
}


void
RNaReplace(DataType *apInput, double aValue) {
    SIMPLE_DISPATCH(apInput->GetPrecision(), basic::NAReplace, *apInput, aValue)
}


DataType *
RGetDiagonal(DataType *apInput) {
    auto precision = apInput->GetPrecision();
    auto output = new DataType(precision);
    SIMPLE_DISPATCH(precision, basic::GetDiagonal, *apInput, *output)
    return output;

}


DataType *
RGetDiagonalWithDims(DataType *apInput, size_t aRow, size_t aCol) {
    auto precision = apInput->GetPrecision();
    auto output = new DataType(precision);
    Dimensions dim(aRow, aCol);
    SIMPLE_DISPATCH(precision, basic::GetDiagonal, *apInput, *output, &dim)
    return output;
}


std::string
RGetType(DataType *apInput) {
    std::string output;
    basic::GetType(*apInput, output);
    return output;
}


DataType *
RGetMin(DataType *apInput) {
    auto precision = apInput->GetPrecision();
    auto output = new DataType(precision);
    size_t index;
    SIMPLE_DISPATCH(precision, basic::MinMax, *apInput, *output, index, false)
    return output;
}


size_t
RGetMinIdx(DataType *apInput) {
    auto precision = apInput->GetPrecision();
    auto output = new DataType(precision);
    size_t index;
    SIMPLE_DISPATCH(precision, basic::MinMax, *apInput, *output, index, false)
    delete output;
    return index;
}


DataType *
RGetMax(DataType *apInput) {
    auto precision = apInput->GetPrecision();
    auto output = new DataType(precision);
    size_t index;
    SIMPLE_DISPATCH(precision, basic::MinMax, *apInput, *output, index, true)
    return output;
}


size_t
RGetMaxIdx(DataType *apInput) {
    auto precision = apInput->GetPrecision();
    auto output = new DataType(precision);
    size_t index;
    SIMPLE_DISPATCH(precision, basic::MinMax, *apInput, *output, index, true)
    delete output;
    return index;
}


DataType *
RSweep(DataType *apInput, DataType *apStats, int aMargin,
       const std::string aOperation) {
    auto precision_a = apInput->GetPrecision();
    auto precision_b = apStats->GetPrecision();
    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto output = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);

    DISPATCHER(operation_comb, basic::Sweep, *apInput, *apStats, *output,
               aMargin, aOperation)
    return output;
}


bool
RIsNa(DataType *apInput, size_t aIdx) {
    return apInput->IsNA(aIdx);
}


size_t
RObjectSize(DataType *apInput) {
    return apInput->GetObjectSize();
}


size_t
RGetNRow(DataType *apInput) {
    return apInput->GetNRow();
}


size_t
RGetNCol(DataType *apInput) {
    return apInput->GetNCol();
}


std::string
RPrint(DataType *apInput) {
    std::string output;
    basic::GetAsStr(*apInput, output);
    return output;
}


DataType *
RGetElementVector(DataType *apInput, size_t aIndex) {
    return apInput->GetElementVector(aIndex);
}


DataType *
RGetElementMatrix(DataType *apInput, size_t aRowIdx,
                  size_t aColIdx) {
    return apInput->GetElementMatrix(aRowIdx, aColIdx);
}