
#include <operations/BasicOperations.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <adapters/RBasicUtilities.hpp>
#include <adapters/RHelpers.hpp>


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
    auto pOutput = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);

    DISPATCHER(operation_comb, basic::ColumnBind, *apInputA, *apInputB,
               *pOutput)
    return pOutput;
}


DataType *
RRBind(DataType *apInputA, DataType *apInputB) {
    auto precision_a = apInputA->GetPrecision();
    auto precision_b = apInputB->GetPrecision();
    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto pOutput = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);

    DISPATCHER(operation_comb, basic::RowBind, *apInputA, *apInputB, *pOutput)
    return pOutput;
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
RReplicate(DataType *apInput, size_t aSize, size_t aLength) {
    if (aLength == 0) {
        aSize = aSize * apInput->GetSize();
    } else {
        aSize = aLength;
    }
    if (aSize == 0) {
        MPR_API_EXCEPTION("Replicate size cannot equal to Zero", -1);
    }
    auto precision = apInput->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, basic::Replicate, *apInput, *pOutput, aSize)
    return pOutput;
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
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, basic::GetDiagonal, *apInput, *pOutput)
    return pOutput;

}


DataType *
RGetDiagonalWithDims(DataType *apInput, size_t aRow, size_t aCol) {
    auto precision = apInput->GetPrecision();
    auto output = new DataType(precision);
    Dimensions dim(aRow, aCol);
    SIMPLE_DISPATCH(precision, basic::GetDiagonal, *apInput, *output, &dim)
    return output;
}


void
RGetType(DataType *apInput) {
    std::string output;
    basic::GetType(*apInput, output);
    Rcpp::Rcout << output;
}


DataType *
RGetMin(DataType *apInput) {
    auto precision = apInput->GetPrecision();
    auto pOutput = new DataType(precision);
    size_t index;
    SIMPLE_DISPATCH(precision, basic::MinMax, *apInput, *pOutput, index, false)
    return pOutput;
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
    auto pOutput = new DataType(precision);
    size_t index;
    SIMPLE_DISPATCH(precision, basic::MinMax, *apInput, *pOutput, index, true)
    return pOutput;
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
    auto pOutput = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);

    DISPATCHER(operation_comb, basic::Sweep, *apInput, *apStats, *pOutput,
               aMargin, aOperation)
    return pOutput;
}


SEXP
RIsNa(DataType *apInput, long aIdx) {

    if (aIdx < 0) {
        Dimensions *pDim = nullptr;
        auto pOutput = apInput->IsNA(pDim);
        if (pDim != nullptr) {
            auto matrix = ToLogicalMatrix(*pOutput, pDim);
            delete pDim;
            return matrix;
        }
        auto vec = ToLogicalVector(*pOutput);
        delete pOutput;
        return vec;
    } else {
        return Rcpp::wrap(apInput->IsNA(aIdx));
    }

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


void
RPrint(DataType *apInput) {
    std::string output;
    basic::GetAsStr(*apInput, output);
    Rcpp::Rcout << output;
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


DataType *
RConcatenate(Rcpp::ListOf <SEXP> aList) {

    std::vector <DataType *> mpr_objects;
    auto list_size = aList.size();
    auto mpr_list_size = list_size;

    /**
     * This if condition is added because Concatenate take 2 MPR object at a
     * time ,so it checks if aList is even or odd and if odd, dummy object
     * will be added.
     **/
    if (list_size % 2 != 0) {
        mpr_list_size++;
    }

    mpr_objects.resize(mpr_list_size);
    size_t i = 0;
    size_t size_out = 0;
    auto precision_out = INT;

    for (auto itr = aList.begin(); itr < aList.end(); ++itr) {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            itr->get());
        if (temp_mpr->IsDataType() && !temp_mpr->IsMatrix()) {
            mpr_objects[ i ] = temp_mpr;
            i++;

            size_out += temp_mpr->GetSize();
            precision_out = GetOutputPrecision(precision_out,
                                               temp_mpr->GetPrecision());
        } else {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure all Objects are MPR Objects and Vectors",
                (int) i);
        }

    }
    /** Add Dummy Object **/
    if (list_size != mpr_list_size) {
        DataType dummy(0, INT);
        mpr_objects[ i ] = &dummy;
    }

    auto pOutput = new DataType(size_out, precision_out);
    auto operation_precision = INT;
    auto precision_one = INT;
    auto precision_two = INT;
    size_t offset = 0;

    for (auto j = 0; j < mpr_list_size; j += 2) {
        precision_one = mpr_objects[ j ]->GetPrecision();
        precision_two = mpr_objects[ j + 1 ]->GetPrecision();
        operation_precision = GetOperationPrecision(precision_one,
                                                    precision_two,
                                                    precision_out);
        DISPATCHER(operation_precision, basic::Concatenate, *mpr_objects[ j ],
                   *mpr_objects[ j + 1 ], *pOutput, offset)
    }
    return pOutput;
}


DataType *
RScale(DataType *apInput, DataType *apCenter, DataType *apScale) {
    auto precision_a = apInput->GetPrecision();
    auto precision_b = apCenter->GetPrecision();
    auto precision_c = apScale->GetPrecision();

    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    output_precision = GetOutputPrecision(output_precision, precision_c);

    auto pOutput = new DataType(output_precision);
    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);

    DISPATCHER(operation_comb, basic::ApplyCenter, *apInput, *apCenter,
               *pOutput)

    operation_comb = GetOperationPrecision(output_precision, precision_c,
                                           output_precision);

    DISPATCHER(operation_comb, basic::ApplyScale, *pOutput, *apScale, *pOutput)
    return pOutput;

}


DataType *
RScale(DataType *apInput, bool aCenter, DataType *apScale) {
    auto precision_a = apInput->GetPrecision();
    auto precision_b = apScale->GetPrecision();

    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto pOutput = new DataType(output_precision);

    auto operation_comb = GetOperationPrecision(precision_a, precision_a,
                                                output_precision);
    DataType dummy_center(precision_b);

    DISPATCHER(operation_comb, basic::ApplyCenter, *apInput, dummy_center,
               *pOutput, &aCenter)

    operation_comb = GetOperationPrecision(output_precision, precision_b,
                                           output_precision);

    DISPATCHER(operation_comb, basic::ApplyScale, *pOutput, *apScale, *pOutput)

    return pOutput;

}


DataType *
RScale(DataType *apInput, DataType *apCenter, bool aScale) {
    auto precision_a = apInput->GetPrecision();
    auto precision_b = apCenter->GetPrecision();

    auto output_precision = GetOutputPrecision(precision_a, precision_b);
    auto pOutput = new DataType(output_precision);

    auto operation_comb = GetOperationPrecision(precision_a, precision_b,
                                                output_precision);
    DataType dummy_scale(precision_b);

    DISPATCHER(operation_comb, basic::ApplyCenter, *apInput, *apCenter,
               *pOutput)

    operation_comb = GetOperationPrecision(output_precision, precision_b,
                                           output_precision);
    DISPATCHER(operation_comb, basic::ApplyScale, *pOutput, dummy_scale,
               *pOutput, &aScale)

    return pOutput;
}


DataType *
RScale(DataType *apInput, bool aCenter, bool aScale) {
    auto precision_a = apInput->GetPrecision();
    auto pOutput = new DataType(precision_a);

    auto operation_comb = GetOperationPrecision(precision_a, precision_a,
                                                precision_a);
    DataType dummy(precision_a);

    DISPATCHER(operation_comb, basic::ApplyCenter, *apInput, dummy,
               *pOutput, &aCenter)

    DISPATCHER(operation_comb, basic::ApplyScale, *pOutput, dummy,
               *pOutput, &aScale)

    return pOutput;
}


DataType *
RScale(DataType *apInput) {
    return RScale(apInput, true, true);
}


DataType *
RScaleDispatcher(SEXP a, SEXP b, SEXP c) {
    auto flag_center = false;
    auto flag_scale = false;
    auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
        a);
    if (!temp_mpr->IsDataType()) {
        MPR_API_EXCEPTION(
            "Undefined Object . Make Sure You're Using MPR Object",
            -1);
    }

    if (TYPEOF(b) == LGLSXP) {
        flag_center = true;
    }
    if (TYPEOF(c) == LGLSXP) {
        flag_scale = true;
    }

    if (flag_center && flag_scale) {
        bool center = Rcpp::as <bool>(b);
        bool scale = Rcpp::as <bool>(c);
        return RScale(temp_mpr, center, scale);

    } else if (flag_center) {
        bool center = Rcpp::as <bool>(b);
        auto temp_scale = (DataType *) Rcpp::internal::as_module_object_internal(
            c);
        if (!temp_scale->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RScale(temp_mpr, center, temp_scale);
    } else if (flag_scale) {
        bool scale = Rcpp::as <bool>(c);
        auto temp_center = (DataType *) Rcpp::internal::as_module_object_internal(
            b);
        if (!temp_center->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RScale(temp_mpr, temp_center, scale);
    } else {
        auto temp_scale = (DataType *) Rcpp::internal::as_module_object_internal(
            c);
        auto temp_center = (DataType *) Rcpp::internal::as_module_object_internal(
            b);
        if (!temp_center->IsDataType() || !temp_scale->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RScale(temp_mpr, temp_center, temp_scale);
    }

}
