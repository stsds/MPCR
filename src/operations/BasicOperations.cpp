/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <operations/BasicOperations.hpp>


using namespace mpcr::operations;
using namespace mpcr::precision;
using namespace mpcr;


template <typename T>
void
basic::MinMax(DataType &aVec, DataType &aOutput, size_t &aMinMaxIdx,
              const bool &aIsMax) {
    if (aVec.GetSize() == 0) {
        return;
    }

    T *pData = (T *) aVec.GetData();
    T *pOutput;
    T min = pData[ 0 ];
    T max = pData[ 0 ];
    size_t min_idx = 0;
    size_t max_idx = 0;
    pOutput = (T *) memory::AllocateArray(1 * sizeof(T), CPU, nullptr);
    auto size = aVec.GetSize();

    for (auto i = 1; i < size; i++) {
        if (!std::isnan(pData[ i ])) {
            if (pData[ i ] < min) {
                min = pData[ i ];
                min_idx = i;
            } else if (pData[ i ] > max) {
                max = pData[ i ];
                max_idx = i;
            }
        }
    }


    if (aIsMax) {
        pOutput[ 0 ] = max;
        aMinMaxIdx = max_idx;
    } else {
        pOutput[ 0 ] = min;
        aMinMaxIdx = min_idx;
    }
    aOutput.ClearUp();
    aOutput.SetSize(1);
    aOutput.SetData((char *) pOutput);
}


void
basic::GetType(DataType &aVec, std::string &aType) {

    std::stringstream ss;
    ss << "MPCR Object : ";

    definitions::Precision temp = aVec.GetPrecision();
    if (temp == definitions::HALF) {
        ss << "16-Bit Precision";
    } else if (temp == definitions::FLOAT) {
        ss << "32-Bit Precision";
    } else if (temp == definitions::DOUBLE) {
        ss << "64-Bit Precision";
    } else {
        MPCR_API_EXCEPTION("Type Error Unknown Type", (int) temp);
    }
    ss << std::endl;
    aType = ss.str();

}


template <typename T>
void
basic::GetDiagonal(DataType &aVec, DataType &aOutput,
                   Dimensions *apDim) {
    Dimensions *pDims;

    if (!aVec.IsMatrix()) {
        if (apDim == nullptr) {
            MPCR_API_EXCEPTION("Matrix Out of Bound No Dimensions is Passed",
                               -1);
        }
        if (!aVec.CanBeMatrix(apDim->GetNRow(), apDim->GetNCol())) {
            MPCR_API_EXCEPTION("Matrix Out of Bound Wrong Dimensions", -1);
        }
        pDims = apDim;
    } else {
        pDims = aVec.GetDimensions();
    }

    aOutput.ClearUp();
    T *pOutput_data;
    T *pData = (T *) aVec.GetData();
    auto count = std::min(pDims->GetNCol(), pDims->GetNRow());
    pOutput_data = (T *) memory::AllocateArray(count * sizeof(T), CPU, nullptr);

    auto row = pDims->GetNRow();

    for (auto i = 0; i < count; i++) {
        pOutput_data[ i ] = pData[ ( i * row ) + i ];
    }

    aOutput.SetSize(count);
    aOutput.SetData((char *) pOutput_data);

}


template <typename T, typename X, typename Y>
void
basic::Sweep(DataType &aVec, DataType &aStats, DataType &aOutput,
             const int &aMargin, const std::string &aFun) {

    aOutput.ClearUp();
    auto rows = aVec.GetNRow();
    auto cols = aVec.GetNCol();

    if (!aVec.IsMatrix()) {
        aOutput.SetSize(aVec.GetSize());
    } else {
        aOutput.ToMatrix(rows, cols);
    }

    T *pInput_data = (T *) aVec.GetData();
    X *pSweep_data = (X *) aStats.GetData();
    Y *pOutput_data;
    size_t idx = 0;


    auto size = aVec.GetSize();
    auto stat_size = aStats.GetSize();
    pOutput_data = (Y *) memory::AllocateArray(size * sizeof(Y), CPU, nullptr);

    if (aMargin == 1 && rows % stat_size ||
        aMargin != 1 && cols % stat_size) {
        MPCR_API_WARN("STATS does not recycle exactly across MARGIN", -1);
    }

    if (aMargin == 1) {
        RUN_OP(pInput_data, pSweep_data, pOutput_data, aFun, stat_size, 0)

    } else {
        size_t counter = 0;
        RUN_OP_COL(pInput_data, pSweep_data, pOutput_data, aFun, stat_size,
                   counter)
    }

    aOutput.SetData((char *) pOutput_data);
}


template <typename T, typename X, typename Y>
void
basic::Concatenate(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                   size_t &aCurrentIdx) {

    if (aCurrentIdx >= aOutput.GetSize()) {
        return;
    }

    if (aInputA.IsMatrix()) {
        MPCR_API_EXCEPTION("Cannot Concatenate a Matrix", -1);
    }

    T *pData_in_one = (T *) aInputA.GetData();
    Y *pData_out = (Y *) aOutput.GetData();
    auto size = aInputA.GetSize();

    std::copy(pData_in_one, pData_in_one + size, pData_out + aCurrentIdx);
    aCurrentIdx += size;

    if (aInputB.GetSize() != 0) {
        if (aInputB.IsMatrix()) {
            MPCR_API_EXCEPTION("Cannot Concatenate a Matrix", -1);
        }

        X *pData_in_two = (X *) aInputB.GetData();
        size = aInputB.GetSize();

        std::copy(pData_in_two, pData_in_two + size, pData_out + aCurrentIdx);
        aCurrentIdx += size;

    }

    aOutput.SetData((char *) pData_out);
}


template <typename T, typename X, typename Y>
void
basic::ColumnBind(DataType &aInputA, DataType &aInputB, DataType &aOutput) {
    if (!aInputA.IsMatrix() || !aInputB.IsMatrix()) {
        MPCR_API_EXCEPTION("Cannot Bind ... Not a Matrix", -1);
    }
    size_t new_size = aInputA.GetSize() + aInputB.GetSize();
    auto dim_one = aInputA.GetDimensions();
    auto dim_two = aInputB.GetDimensions();
    if (dim_one->GetNRow() != dim_two->GetNRow()) {
        MPCR_API_EXCEPTION("Cannot Bind ... Different Row Size", -1);
    }

    size_t num_rows = dim_one->GetNRow();
    size_t num_cols = dim_one->GetNCol() + dim_two->GetNCol();

    T *pData_one = (T *) aInputA.GetData();
    X *pData_two = (X *) aInputB.GetData();
    Y *pData_out = (Y *) memory::AllocateArray(new_size * sizeof(Y), CPU,
                                               nullptr);

    std::copy(pData_one, pData_one + aInputA.GetSize(), pData_out);
    std::copy(pData_two, pData_two + aInputB.GetSize(),
              pData_out + aInputA.GetSize());

    aOutput.ClearUp();
    aOutput.ToMatrix(num_rows, num_cols);
    aOutput.SetData((char *) pData_out);


}


template <typename T, typename X, typename Y>
void
basic::RowBind(DataType &aInputA, DataType &aInputB, DataType &aOutput) {
    if (!aInputA.IsMatrix() || !aInputB.IsMatrix()) {
        MPCR_API_EXCEPTION("Cannot Bind ... Not a Matrix", -1);
    }
    size_t new_size = aInputA.GetSize() + aInputB.GetSize();
    auto dim_one = aInputA.GetDimensions();
    auto dim_two = aInputB.GetDimensions();
    if (dim_one->GetNCol() != dim_two->GetNCol()) {
        MPCR_API_EXCEPTION("Cannot Bind ... Different Column Size", -1);
    }

    size_t num_cols = dim_one->GetNCol();
    size_t num_rows_in_1 = dim_one->GetNRow();
    size_t num_rows_in_2 = dim_two->GetNRow();
    size_t num_rows = num_rows_in_1 + num_rows_in_2;
    T *pData_one = (T *) aInputA.GetData();
    X *pData_two = (X *) aInputB.GetData();
    Y *pData_out = (Y *) memory::AllocateArray(new_size * sizeof(Y), CPU,
                                               nullptr);
    size_t offset;
    size_t offset_one;
    size_t offset_two;

    for (auto i = 0; i < num_cols; ++i) {
        offset_one = i * num_rows_in_1;
        offset_two = i * num_rows_in_2;
        offset = i * num_rows;
        std::copy(pData_one + offset_one,
                  pData_one + offset_one + num_rows_in_1,
                  pData_out + offset);
        offset += num_rows_in_1;
        std::copy(pData_two + offset_two,
                  pData_two + offset_two + num_rows_in_2,
                  pData_out + offset);
    }

    aOutput.ClearUp();
    aOutput.ToMatrix(num_rows, num_cols);
    aOutput.SetData((char *) pData_out);
}


bool
basic::IsDouble(DataType &aInput) {
    return ( aInput.GetPrecision() == DOUBLE );
}


bool
basic::IsFloat(DataType &aInput) {
    return ( aInput.GetPrecision() == FLOAT );
}


bool
basic::IsSFloat(DataType &aInput) {
    return ( aInput.GetPrecision() == HALF );
}


template <typename T>
void
basic::Replicate(DataType &aInput, DataType &aOutput, const size_t &aSize) {

    T *pData = (T *) aInput.GetData();
    T *pBuffer = (T *) memory::AllocateArray(aSize * sizeof(T), CPU, nullptr);
    size_t data_size = aInput.GetSize();
    for (auto i = 0; i < aSize; ++i) {
        pBuffer[ i ] = pData[ i % data_size ];
    }

    aOutput.ClearUp();
    aOutput.SetSize(aSize);
    aOutput.SetData((char *) pBuffer);

}


void
basic::GetAsStr(DataType &aVec, std::string &aType) {
    GetType(aVec, aType);
    std::stringstream ss;
    ss << std::endl;
    if (aVec.IsMatrix()) {
        ss << "Matrix Of Dimensions :";
        auto dim = aVec.GetDimensions();
        ss << std::endl;
        ss << "Number of Rows = " << dim->GetNRow() << std::endl;
        ss << "Number of Column = " << dim->GetNCol() << std::endl;
    } else {
        ss << "Vector Of Size :" << aVec.GetSize() << std::endl;
        auto itr = ( 10 > aVec.GetSize()) ? aVec.GetSize() : 10;
        ss << "Data :" << std::endl << std::left << std::setfill(' ')
           << std::setw(3) << "[ ";
        for (auto i = 0; i < itr; ++i) {
            ss << aVec.GetVal(i) << "   ";
        }
        ss << " ... ]" << std::endl;

    }
    aType += ss.str();
}


template <typename T>
void
basic::NAExclude(DataType &aInputA) {

    T *pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto counter = size;
    if (aInputA.IsMatrix()) {

        std::unordered_set <size_t> row_idx;
        std::vector <size_t> row_idx_na;
        auto rows = aInputA.GetNRow();
        auto cols = aInputA.GetNCol();
        for (auto i = 0; i < cols; i++) {
            for (auto j = 0; j < rows; j++) {
                if (std::isnan(pData[ ( i * rows ) + j ])) {
                    row_idx_na.push_back(j);
                } else {
                    row_idx.insert(j);
                }
            }
        }

        for (auto &x: row_idx_na) {
            row_idx.erase(x);
        }

        counter = row_idx.size() * cols;
        T *pOutput = (T *) memory::AllocateArray(counter * sizeof(T), CPU,
                                                 nullptr);
        aInputA.SetSize(counter);
        auto row_size_new = row_idx.size();
        aInputA.SetDimensions(row_size_new, cols);
        counter = 0;
        size_t idx;


        for (auto i = 0; i < cols; i++) {
            for (auto j = 0; j < rows; j++) {
                idx = ( i * rows ) + j;
                if (row_idx.find(j) != row_idx.end()) {
                    pOutput[ counter ] = pData[ idx ];
                    counter++;
                }
            }
        }

        aInputA.SetData((char *) pOutput);


    } else {
        for (auto i = 0; i < size; i++) {
            counter -= std::isnan(pData[ i ]);
        }
        if (counter == size) {
            return;
        }
        T *pOutput = (T *) memory::AllocateArray(counter * sizeof(T), CPU,
                                                 nullptr);
        aInputA.SetSize(counter);
        counter = 0;
        for (auto i = 0; i < size; ++i) {
            if (!std::isnan(pData[ i ])) {
                pOutput[ counter++ ] = pData[ i ];
            }
        }

        aInputA.SetData((char *) pOutput);
    }

}


template <typename T, typename X, typename Y>
void
basic::ApplyCenter(DataType &aInputA, DataType &aCenter, DataType &aOutput,
                   const bool *apCenter) {
    auto pData_input = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto col = aInputA.GetNCol();
    auto row = aInputA.GetNRow();
    aOutput.ClearUp();
    aOutput.SetSize(size);
    aOutput.SetDimensions(row, col);
    auto pOutput = (Y *) memory::AllocateArray(size * sizeof(Y), CPU, nullptr);
    size_t start_idx;

    if (apCenter != nullptr) {
        if (*apCenter) {

            double accum;
            size_t counter;
            for (auto i = 0; i < row; i++) {
                accum = 0;
                counter = 0;
                for (auto j = 0; j < col; j++) {
                    start_idx = ( j * row ) + i;
                    auto element = pData_input[ start_idx ];
                    if (!std::isnan(element)) {
                        accum += element;
                        counter++;
                    }

                }
                accum = accum / counter;
                for (auto j = 0; j < col; j++) {
                    start_idx = ( j * row ) + i;
                    pOutput[ start_idx ] = pData_input[ start_idx ] - accum;
                }
            }

        } else {
            //no centering is done
            std::copy(pData_input, pData_input + size, pOutput);
        }
    } else {
        //subtract col element from its respective element in aCenter
        auto pData_center = (X *) aCenter.GetData();
        auto center_size = aCenter.GetSize();
        if (col != center_size) {
            MPCR_API_EXCEPTION(
                "Cannot Center with the Provided Data, Column size doesn't equal Center Vector Size",
                -1);
        }
        auto data_size = aInputA.GetSize();
        for (auto i = 0; i < data_size; i++) {
            pOutput[ i ] = pData_input[ i ] - pData_center[ i % center_size ];
        }
    }

    aOutput.SetData((char *) pOutput);
}


template <typename T, typename X, typename Y>
void
basic::ApplyScale(DataType &aInputA, DataType &aScale, DataType &aOutput,
                  const bool *apScale) {

    auto pData_input = (T *) aInputA.GetData();
    auto pOutput = (Y *) aOutput.GetData();

    if (apScale != nullptr) {
        if (*apScale) {
            auto col_size = aInputA.GetNCol();
            auto row_size = aInputA.GetNRow();
            size_t counter;
            size_t start_idx;
            double mean;
            double accum;
            double variance;
            double stdev;

            for (auto i = 0; i < row_size; i++) {
                accum = 0;
                counter = 0;
                variance = 0.0;
                for (auto j = 0; j < col_size; j++) {
                    start_idx = ( j * row_size ) + i;
                    auto element = pData_input[ start_idx ];
                    if (!std::isnan(element)) {
                        accum += element;
                        counter++;
                    }
                }
                mean = accum / counter;
                for (auto j = 0; j < col_size; j++) {
                    start_idx = ( j * row_size ) + i;
                    auto element = pData_input[ start_idx ];
                    if (!std::isnan(element)) {
                        variance += ( element - mean ) * ( element - mean );
                    }
                }


                stdev = sqrt(variance / ( col_size - 1 ));

                for (auto j = 0; j < col_size; j++) {
                    start_idx = ( j * row_size ) + i;
                    pOutput[ start_idx ] = pOutput[ start_idx ] / stdev;
                }
            }
        }
    } else {
        auto pData_scale = (X *) aScale.GetData();
        auto scale_size = aScale.GetSize();
        auto col_size = aInputA.GetNCol();
        if (col_size != scale_size) {
            MPCR_API_EXCEPTION(
                "Cannot Scale with the Provided Data, Column size doesn't equal Scale Vector Size",
                -1);
        }
        auto data_size = aInputA.GetSize();
        for (auto i = 0; i < data_size; i++) {
            pOutput[ i ] = pOutput[ i ] / pData_scale[ i % scale_size ];
        }

    }

    aOutput.SetData((char *) pOutput);


}


template <typename T>
void
basic::NAReplace(DataType &aInputA, const double &aValue) {
    T *pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    for (auto i = 0; i < size; i++) {
        if (std::isnan(pData[ i ])) {
            pData[ i ] = (T) aValue;
        }
    }

    aInputA.SetData((char *) pData);

}


INSTANTIATE(void, basic::ColumnBind, DataType &aInputA, DataType &aInputB,
            DataType &aOutput)

INSTANTIATE(void, basic::RowBind, DataType &aInputA, DataType &aInputB,
            DataType &aOutput)

INSTANTIATE(void, basic::Sweep, DataType &aVec, DataType &aStats,
            DataType &aOutput,
            const int &aMargin, const std::string &aFun)

INSTANTIATE(void, basic::ApplyCenter, DataType &aInputA, DataType &aCenter,
            DataType &aOutput, const bool *apCenter)

INSTANTIATE(void, basic::ApplyScale, DataType &aInputA, DataType &aScale,
            DataType &aOutput, const bool *apScale)

INSTANTIATE(void, basic::Concatenate, DataType &aInputA, DataType &aInputB,
            DataType &aOutput,
            size_t &aCurrentIdx)

SIMPLE_INSTANTIATE(void, basic::GetDiagonal, DataType &aVec, DataType &aOutput,
                   Dimensions *apDim)

SIMPLE_INSTANTIATE(void, basic::MinMax, DataType &aVec, DataType &aOutput,
                   size_t &aMinMaxIdx, const bool &aIsMax)

SIMPLE_INSTANTIATE(void, basic::Replicate, DataType &aInput, DataType &aOutput,
                   const size_t &aSize)

SIMPLE_INSTANTIATE(void, basic::NAExclude, DataType &aInputA)

SIMPLE_INSTANTIATE(void, basic::NAReplace, DataType &aInputA,
                   const double &aValue)
