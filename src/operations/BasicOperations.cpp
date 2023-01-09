

#include <operations/BasicOperations.hpp>
#include <utilities/MPRErrorHandler.hpp>
#include <utilities/MPRDispatcher.hpp>


using namespace mpr::operations;
using namespace mpr::precision;


template <typename T>
void
basic::MinMax(DataType &aVec, DataType &aOutput, size_t &aMinMaxIdx,
              const bool &aIsMax) {
    if (aVec.GetSize() == 0) {
        return;
    }

    T *data = (T *) aVec.GetData();
    T *output;
    T min = data[ 0 ];
    T max = data[ 0 ];
    size_t MinIdx = 0;
    size_t MaxIdx = 0;
    output = new T[1];
    auto size = aVec.GetSize();

    for (auto i = 1; i < size; i++) {
        if (!std::isnan(data[ i ])) {
            if (data[ i ] < min) {
                min = data[ i ];
                MinIdx = i;
            } else if (data[ i ] > max) {
                max = data[ i ];
                MaxIdx = i;
            }
        }
    }


    if (aIsMax) {
        output[ 0 ] = max;
        aMinMaxIdx = MaxIdx;
    } else {
        output[ 0 ] = min;
        aMinMaxIdx = MinIdx;
    }
    aOutput.ClearUp();
    aOutput.SetSize(1);
    aOutput.SetData((char *) output);
}


void
basic::GetType(DataType &aVec, std::string &aType) {

    std::stringstream ss;
    ss << "MPR Object : ";

    precision::Precision temp = aVec.GetPrecision();
    if (temp == precision::INT) {
        ss << "16-Bit Precision";
    } else if (temp == precision::FLOAT) {
        ss << "32-Bit Precision";
    } else if (temp == precision::DOUBLE) {
        ss << "64-Bit Precision";
    } else {
        MPR_API_EXCEPTION("Type Error Unknown Type", (int) temp);
    }
    ss << std::endl;
    aType = ss.str();

}


template <typename T>
void
basic::GetDiagonal(DataType &aVec, DataType &aOutput,
                   Dimensions *apDim) {
    Dimensions *dims;

    if (!aVec.IsMatrix()) {
        if (apDim == nullptr) {
            MPR_API_EXCEPTION("Matrix Out of Bound No Dimensions is Passed",
                              -1);
        }
        if (!aVec.CanBeMatrix(apDim->GetNRow(), apDim->GetNCol())) {
            MPR_API_EXCEPTION("Matrix Out of Bound Wrong Dimensions", -1);
        }
        dims = apDim;
    } else {
        dims = aVec.GetDimensions();
    }

    aOutput.ClearUp();
    T *output_data;
    T *data = (T *) aVec.GetData();
    auto col = dims->GetNCol();
    output_data = new T[col];

    for (auto i = 0; i < col; i++) {
        output_data[ i ] = data[ ( i * col ) + i ];
    }

    aOutput.SetSize(col);
    aOutput.SetData((char *) output_data);

}


template <typename T, typename X, typename Y>
void
basic::Sweep(DataType &aVec, DataType &aStats, DataType &aOutput,
             const int &aMargin, const std::string &aFun) {
    aOutput.ClearUp();
    auto row = aVec.GetNRow();
    auto col = aVec.GetNCol();
    aOutput.SetSize(row * col);
    aOutput.SetDimensions(row, col);
    T *input_data = (T *) aVec.GetData();
    X *sweep_data = (X *) aStats.GetData();
    Y *output_data;
    size_t idx;


    auto size = aVec.GetSize();
    output_data = new Y[size];

    if (aMargin == 1 && row % aStats.GetSize() ||
        aMargin != 1 && col % aStats.GetSize()) {
        MPR_API_WARN("STATS does not recycle exactly across MARGIN", -1);
    }

    if (aMargin == 1) {
        RUN_OP(input_data, sweep_data, output_data, aFun, col)

    } else {
        RUN_OP(input_data, sweep_data, output_data, aFun, row)
    }
    aOutput.SetData((char *) output_data);
}


template <typename T, typename X, typename Y>
void
basic::Concatenate(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                   size_t &aCurrentIdx) {

    if (aCurrentIdx >= aOutput.GetSize()) {
        return;
    }

    T *data_in_one = (T *) aOutput.GetData();
    Y *data_out = (Y *) aOutput.GetData();
    auto size = aInputA.GetSize();
    std::copy(data_in_one, data_in_one + size, data_out + aCurrentIdx);
    aCurrentIdx += size;

    if (aInputB.GetSize() != 0) {
        X *data_in_two = (X *) aOutput.GetData();
        size = aInputB.GetSize();
        std::copy(data_in_two, data_in_two + size, data_out + aCurrentIdx);
        aCurrentIdx += size;
    }

    aOutput.SetData((char *) data_out);
}


template <typename T, typename X, typename Y>
void
basic::ColumnBind(DataType &aInputA, DataType &aInputB, DataType &aOutput) {
    if (!aInputA.IsMatrix() || !aInputB.IsMatrix()) {
        MPR_API_EXCEPTION("Cannot Bind ... Not a Matrix", -1);
    }
    size_t new_size = aInputA.GetSize() + aInputB.GetSize();
    auto dim_one = aInputA.GetDimensions();
    auto dim_two = aInputB.GetDimensions();
    if (dim_one->GetNRow() != dim_two->GetNRow()) {
        MPR_API_EXCEPTION("Cannot Bind ... Different Row Size", -1);
    }
    size_t num_rows = dim_one->GetNRow();
    size_t num_cols = dim_one->GetNCol() + dim_two->GetNCol();
    size_t num_cols_in_1 = dim_one->GetNCol();
    size_t num_cols_in_2 = dim_two->GetNCol();
    T *data_one = (T *) aInputA.GetData();
    X *data_two = (X *) aInputB.GetData();
    Y *data_out = new Y[new_size];
    size_t offset;
    size_t offset_one;
    size_t offset_two;
    for (auto i = 0; i < num_rows; ++i) {
        offset_one = i * num_cols_in_1;
        offset_two = i * num_cols_in_2;
        offset = i * num_cols;
        std::copy(data_one + offset_one, data_one + offset_one + num_cols_in_1,
                  data_out + offset);
        offset += num_cols_in_1;
        std::copy(data_two + offset_two, data_two + offset_two + num_cols_in_2,
                  data_out + offset);
    }
    aOutput.ClearUp();
    aOutput.ToMatrix(num_rows, num_cols);
    aOutput.SetData((char *) data_out);
}


template <typename T, typename X, typename Y>
void
basic::RowBind(DataType &aInputA, DataType &aInputB, DataType &aOutput) {
    if (!aInputA.IsMatrix() || !aInputB.IsMatrix()) {
        MPR_API_EXCEPTION("Cannot Bind ... Not a Matrix", -1);
    }
    size_t new_size = aInputA.GetSize() + aInputB.GetSize();
    auto dim_one = aInputA.GetDimensions();
    auto dim_two = aInputB.GetDimensions();
    if (dim_one->GetNCol() != dim_two->GetNCol()) {
        MPR_API_EXCEPTION("Cannot Bind ... Different Row Size", -1);
    }
    size_t num_rows = dim_one->GetNRow() + dim_two->GetNRow();
    size_t num_cols = dim_one->GetNCol();

    T *data_one = (T *) aInputA.GetData();
    X *data_two = (X *) aInputB.GetData();
    Y *data_out = new Y[new_size];

    /** Check if indexing needs +1 **/
    std::copy(data_one, data_one + aInputA.GetSize(), data_out);
    std::copy(data_two, data_two + aInputB.GetSize(),
              data_out + aInputA.GetSize());

    aOutput.ClearUp();
    aOutput.ToMatrix(num_rows, num_cols);
    aOutput.SetData((char *) data_out);
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
    return ( aInput.GetPrecision() == INT );
}


template <typename T>
void
basic::Replicate(DataType &aInput, DataType &aOutput, const size_t &aSize) {

    T *data = (T *) aInput.GetData();
    T *buffer = new T[aSize];
    size_t data_size = aInput.GetSize();
    for (auto i = 0; i < aSize; ++i) {
        buffer[ i ] = data[ i % data_size ];
    }

    aOutput.ClearUp();
    aOutput.SetSize(aSize);
    aOutput.SetData((char *) buffer);

}


void
basic::GetAsStr(DataType &aVec, std::string &aType) {
    GetType(aVec, aType);
    std::stringstream ss;
    ss << std::endl;
    if (aVec.IsMatrix()) {
        ss << "Matrix Of Dimensions :";
        auto dim = aVec.GetDimensions();
        ss << "Number of Rows = " << dim->GetNRow() << std::endl;
        ss << "Number of Column = " << dim->GetNCol() << std::endl;
    } else {
        ss << "Vector Of Size :" << aVec.GetSize() << std::endl;
    }
    auto itr = ( 10 > aVec.GetSize()) ? aVec.GetSize() : 10;
    ss << "Data :" << std::endl << std::left << std::setfill(' ')
       << std::setw(10) << "[ ";
    for (auto i = 0; i < itr; ++i) {
        ss << aVec.GetVal(i) << "\t";
    }
    ss << " ... " << std::endl;
    aType += ss.str();
}


template <typename T>
void basic::NAExclude(DataType &aInputA) {

    T *data = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto counter = size;
    if (aInputA.IsMatrix()) {
        //TODO: check how matrix should be excluded
    } else {
        for (auto i = 0; i < size; i++) {
            counter -= std::isnan(data[ i ]);
        }
        if (counter == size) {
            return;
        }
        T *output = new T[counter];
        aInputA.SetSize(counter);
        counter = 0;
        for (auto i = 0; i < size; ++i) {
            if (!std::isnan(data[ i ])) {
                output[ counter++ ] = data[ i ];
            }
        }

        aInputA.SetData((char *) output);
    }

}


template <typename T, typename X, typename Y>
void basic::ApplyScale() {
    //not implemented yet
}


template <typename T>
void
basic::NAReplace(DataType &aInputA, const double &aValue) {
    T *data = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    for (auto i = 0; i < size; i++) {
        if (std::isnan(data[ i ])) {
            data[ i ] = (T) aValue;
        }
    }

}


INSTANTIATE(void, basic::ColumnBind, DataType &aInputA, DataType &aInputB,
            DataType &aOutput)

INSTANTIATE(void, basic::RowBind, DataType &aInputA, DataType &aInputB,
            DataType &aOutput)

INSTANTIATE(void, basic::Sweep, DataType &aVec, DataType &aStats,
            DataType &aOutput,
            const int &aMargin, const std::string &aFun)

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