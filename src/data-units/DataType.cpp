/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <data-units/DataType.hpp>
#include <utilities/MPCRDispatcher.hpp>
#include <adapters/RBinaryOperations.hpp>
#include <Rcpp.h>


using namespace mpcr::precision;


DataType::DataType(size_t aSize, Precision aPrecision) {
    this->SetMagicNumber();
    this->mpData = nullptr;
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mSize = aSize;
    this->mpDimensions = nullptr;
    this->mMatrix = false;
    SIMPLE_DISPATCH(this->mPrecision, Init)
}


DataType::DataType(std::vector <double> aValues, std::string aPrecision) {

    this->SetMagicNumber();
    this->mpData = nullptr;
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mSize = aValues.size();
    this->mpDimensions = nullptr;
    this->mMatrix = false;
    SIMPLE_DISPATCH(this->mPrecision, Init, &aValues)

}


DataType::DataType(std::vector <double> &aValues, const size_t &aRow,
                   const size_t &aCol, const std::string &aPrecision) {
    this->SetMagicNumber();
    this->mpData = nullptr;
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mSize = aValues.size();
    this->mpDimensions = new Dimensions(aRow, aCol);
    this->mMatrix = true;
    SIMPLE_DISPATCH(this->mPrecision, Init, &aValues)
}


DataType::DataType(std::vector <double> &aValues,
                   mpcr::precision::Precision aPrecision) {
    this->SetMagicNumber();
    this->mpData = nullptr;
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mSize = aValues.size();
    this->mpDimensions = nullptr;
    this->mMatrix = false;
    SIMPLE_DISPATCH(this->mPrecision, Init, &aValues)
}


DataType::DataType(size_t aSize, int aPrecision) {
    this->SetMagicNumber();
    this->mpData = nullptr;
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mpDimensions = nullptr;
    this->mMatrix = false;
    this->mSize = aSize;
    SIMPLE_DISPATCH(this->mPrecision, Init)
}


DataType::DataType(size_t aSize, const std::string &aPrecision) {
    this->SetMagicNumber();
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mpData = nullptr;
    this->mpDimensions = nullptr;
    this->mMatrix = false;
    this->mSize = aSize;
    SIMPLE_DISPATCH(this->mPrecision, Init)

}


DataType::DataType(size_t aRow, size_t aCol, Precision aPrecision) {
    this->SetMagicNumber();
    this->mpData = nullptr;
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mpDimensions = new Dimensions(aRow, aCol);
    this->mMatrix = true;
    this->mSize = aRow * aCol;
    SIMPLE_DISPATCH(this->mPrecision, Init)
}


DataType::DataType(mpcr::precision::Precision aPrecision) {
    this->SetMagicNumber();
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mMatrix = false;
    this->mpDimensions = nullptr;
    this->mSize = 0;
    this->mpData = nullptr;
}


DataType::DataType(const DataType &aDataType) {
    this->SetMagicNumber();
    this->mpData = nullptr;
    this->mpDimensions = nullptr;
    this->mSize = aDataType.mSize;
    this->mPrecision = aDataType.mPrecision;
    this->mMatrix = aDataType.mMatrix;
    if (this->mMatrix) {
        this->mpDimensions = new Dimensions(*aDataType.GetDimensions());
    }
    if (this->mSize != 0) {
        SIMPLE_DISPATCH(this->mPrecision, GetCopyOfData, aDataType.mpData,
                        this->mpData)
    }
}


DataType::DataType(DataType &aDataType,
                   const mpcr::precision::Precision &aPrecision) {
    this->SetMagicNumber();
    this->mpData = nullptr;
    this->mpDimensions = nullptr;
    this->mSize = aDataType.mSize;
    this->mPrecision = aPrecision;
    this->mMatrix = aDataType.mMatrix;
    if (this->mMatrix) {
        this->mpDimensions = new Dimensions(*aDataType.GetDimensions());
    }
    if (this->mSize != 0) {
        auto precision = GetOperationPrecision(aDataType.mPrecision,
                                               this->mPrecision,
                                               DOUBLE);
        DISPATCHER(precision, DataType::GetCopyOfData, aDataType, *this)
    }
}


DataType::~DataType() {
    delete[] mpData;
    delete mpDimensions;
}


template <typename T>
void
DataType::Init(std::vector <double> *aValues) {
    if (this->mSize == 0) {
        return;
    }
    double val = 0;
#ifdef RUNNING_CPP
    val = 1.5;
#endif
    bool flag = ( aValues == nullptr );
    T *temp = new T[mSize];
    for (auto i = 0; i < mSize; i++) {
        if (flag) {
            temp[ i ] = (T) val;
        } else {
            temp[ i ] = (T) aValues->at(i);
        }
    }
    this->mpData = (char *) temp;

}


std::string
DataType::PrintRow(const size_t &aRowIdx) {

    if (aRowIdx > this->GetNRow()) {
        MPCR_API_EXCEPTION("Segmentation fault index out of Bound", -1);
    }
    std::stringstream ss;
    SIMPLE_DISPATCH(this->mPrecision, DataType::PrintRowsDispatcher, aRowIdx,
                    ss)
    return ss.str();

}


template <typename T>
void
DataType::PrintRowsDispatcher(const size_t &aRowIdx,
                              std::stringstream &aRowAsString) {

    auto pData = (T *) this->mpData;
    auto col = GetNCol();
    auto row = GetNRow();
    size_t idx = 0;
    auto temp_col = col > 16 ? 16 : col;

    for (auto i = 0; i < temp_col; i++) {
        idx = ( i * row ) + aRowIdx;
        aRowAsString << std::setfill(' ') << std::setw(14)
                     << std::setprecision(7) << pData[ idx ] << "\t";
    }
}


template <typename T>
void
DataType::PrintVal() {
    std::stringstream ss;
    auto stream_size = 10000;
    T *temp = (T *) this->mpData;

    if (this->mMatrix) {
        auto rows = this->mpDimensions->GetNRow();
        auto cols = this->mpDimensions->GetNCol();
        ss << "Precision  : " << GetPrecisionAsString(this->mPrecision)
           << "  Precision " << std::endl;
        ss << "Number of Rows : " << rows << std::endl;
        ss << "Number of Columns : " << cols << std::endl;
        ss << "---------------------" << std::endl;
        size_t start_idx;
        size_t print_col = ( cols > 16 ) ? 16 : cols;
        size_t print_rows = ( rows > 100 ) ? 100 : rows;

        for (auto i = 0; i < print_rows; i++) {
            ss << " [\t";
            for (auto j = 0; j < print_col; j++) {
                start_idx = ( j * rows ) + i;
                ss << std::setfill(' ') << std::setw(14) << std::setprecision(7)
                   << temp[ start_idx ] << "\t";
            }
            ss << std::setfill(' ') << std::setw(14) << "]" << std::endl;
            if (ss.gcount() > stream_size) {
#ifdef RUNNING_CPP
                std::cout << std::string(ss.str());
#endif

#ifndef RUNNING_CPP
                Rcpp::Rcout << std::string(ss.str());
#endif
                ss.clear();
            }
        }
        if (print_rows * print_col != this->mSize) {
            ss << "Note Only Matrix with size 100*13 is printed" <<
               std::endl;
        }
#ifdef RUNNING_CPP
        std::cout << std::string(ss.str());
#endif

#ifndef RUNNING_CPP
        Rcpp::Rcout << std::string(ss.str());
#endif

    } else {
        ss << "Vector Size : " << mSize <<
           std::endl;
        ss << "---------------------" <<
           std::endl;
        auto counter_rows = 0;
        for (auto i = 0; i < mSize; i++) {
            if (i % 7 == 0) {
                ss << std::endl;
                ss << "[ " << counter_rows + 1 << " ]" << "\t";
                counter_rows += 7;
            }
            ss << std::setfill(' ') << std::setw(14) << std::setprecision(7)
               << temp[ i ];
            if (i % 100 == 0) {
                if (ss.gcount() > stream_size) {
#ifdef RUNNING_CPP
                    std::cout << std::string(ss.str());
#endif

#ifndef RUNNING_CPP
                    Rcpp::Rcout << std::string(ss.str());
#endif
                    ss.clear();
                }
            }
        }
        ss << std::endl;
#ifdef RUNNING_CPP
        std::cout << std::string(ss.str());
#endif

#ifndef RUNNING_CPP
        Rcpp::Rcout << std::string(ss.str());
#endif
    }

}


void
DataType::Print() {
    SIMPLE_DISPATCH(mPrecision, PrintVal)
}


Precision &
DataType::GetPrecision() {
    return this->mPrecision;
}


char *
DataType::GetData() {
    return this->mpData;
}


size_t
DataType::GetSize() const {
    return this->mSize;
}


template <typename T>
void
DataType::GetValue(size_t aIndex, double &aOutput) {
    aOutput = (double) (((T *) this->mpData )[ aIndex ] );
}


double
DataType::GetVal(size_t aIndex) {
    double temp=0;
    if (aIndex >= this->mSize) {
        MPCR_API_EXCEPTION("Segmentation Fault Index Out Of Bound", -1);
    }
    SIMPLE_DISPATCH(mPrecision, GetValue, aIndex, temp)
    return temp;
}


template <typename T>
void
DataType::SetValue(size_t aIndex, double &aVal) {

    T *data = (T *) this->mpData;
    data[ aIndex ] = (T) aVal;
}


void
DataType::SetVal(size_t aIndex, double aVal) {
    if (aIndex >= this->mSize) {
        MPCR_API_EXCEPTION("Segmentation Fault Index Out Of Bound", -1);
    }
    SIMPLE_DISPATCH(mPrecision, SetValue, aIndex, aVal)

}


void
DataType::SetPrecision(mpcr::precision::Precision aPrecision) {
    this->ClearUp();
    this->mPrecision = aPrecision;
}


void
DataType::ToMatrix(size_t aRow, size_t aCol) {
    this->mpDimensions = new Dimensions(aRow, aCol);
    this->mSize = aRow * aCol;
    this->mMatrix = true;
}


bool
DataType::IsMatrix() const {
    return this->mMatrix;
}


void
DataType::ToVector() {
    if (this->mpDimensions != nullptr) {
        delete this->mpDimensions;
        this->mpDimensions = nullptr;
        this->mMatrix = false;
    }
}


size_t
DataType::GetMatrixIndex(size_t aRow, size_t aCol) {
    if (!this->mMatrix) {
        MPCR_API_EXCEPTION("Not a Matrix Fault.", -1);
    }
    if (aRow >= mpDimensions->GetNRow() || aCol >= mpDimensions->GetNCol() ||
        aRow < 0 || aCol < 0) {
        MPCR_API_EXCEPTION("Segmentation Fault Index Out Of Bound", -1);
    }

    return ( aCol * mpDimensions->GetNRow()) + aRow;
}


void
DataType::SetData(char *aData) {
    if (aData != mpData) {
        delete[] mpData;
    }
    this->mpData = aData;
}


void
DataType::SetSize(size_t aSize) {
    this->mSize = aSize;
}


size_t
DataType::GetNRow() const {
    if (mMatrix) {
        return this->mpDimensions->GetNRow();
    }
    if (this->mSize == 0) {
        return 0;
    }
    return 1;
}


size_t
DataType::GetNCol() const {
    if (mMatrix) {
        return this->mpDimensions->GetNCol();
    }
    if (this->mSize == 0) {
        return 0;
    }
    return this->mSize;
}


void
DataType::SetDimensions(size_t aRow, size_t aCol) {

    size_t size = aRow * aCol;
    if (size != this->mSize) {
        MPCR_API_EXCEPTION("Segmentation Fault Matrix Out Of Bound", -1);
    }
    this->mSize = size;
    if (this->mpDimensions != nullptr) {
        this->mpDimensions->SetNRow(aRow);
        this->mpDimensions->SetNCol(aCol);
    } else {
        this->mMatrix = true;
        this->mpDimensions = new Dimensions(aRow, aCol);
    }

}


Dimensions *
DataType::GetDimensions() const {
    return this->mpDimensions;
}


template <typename T>
void
DataType::GetCopyOfData(const char *apSrc, char *&apDest) {
    T *data = (T *) apSrc;
    auto size = this->mSize;
    T *pOutput = new T[size];


    memcpy((char *) pOutput, (char *) data, size * sizeof(T));
    apDest = (char *) pOutput;
}


template <typename T, typename X, typename Y>
void
DataType::GetCopyOfData(DataType &aSrc, DataType &aDestination) {
    T *data = (T *) aSrc.GetData();
    auto size = aDestination.mSize;
    X *pOutput = new X[size];

    std::copy(data, data + size, pOutput);
    aDestination.SetData((char *) pOutput);

}


DataType &
DataType::operator =(const DataType &aDataType) {
    this->mSize = aDataType.mSize;
    this->mPrecision = aDataType.mPrecision;
    this->mMatrix = aDataType.mMatrix;
    this->mpData = nullptr;
    if (this->mMatrix) {
        this->mpDimensions = new Dimensions(*aDataType.GetDimensions());
    } else {
        this->mpDimensions = nullptr;
    }

    if (this->mSize != 0) {
        SIMPLE_DISPATCH(this->mPrecision, GetCopyOfData, aDataType.mpData,
                        this->mpData)
    }
    return *this;
}


bool
DataType::IsNA(const size_t &aIndex) {
    bool flag = false;
    SIMPLE_DISPATCH(this->mPrecision, CheckNA, aIndex, flag)
    return flag;
}


template <typename T>
void
DataType::CheckNA(const size_t &aIndex, bool &aFlag) {
    T *data = (T *) this->mpData;
    aFlag = std::isnan(data[ aIndex ]);
}


template <typename T>
void
DataType::GetDataSize(size_t &aDataSize) {
    aDataSize = this->mSize * sizeof(T);
}


size_t
DataType::GetObjectSize() {
    size_t data_size;
    SIMPLE_DISPATCH(this->mPrecision, GetDataSize, data_size)
    if (this->mMatrix) {
        data_size += 3 * sizeof(size_t);
    } else {
        data_size += sizeof(size_t);
    }
    data_size += sizeof(bool);
    data_size += sizeof(Precision);
    return data_size;
}


double
DataType::GetValMatrix(const size_t &aRow, const size_t &aCol) {
    auto idx = this->GetMatrixIndex(aRow, aCol);
    return GetVal(idx);
}


void
DataType::SetValMatrix(size_t aRow, size_t aCol, double aVal) {
    auto idx = this->GetMatrixIndex(aRow, aCol);
    SetVal(idx, aVal);
}


void DataType::SetMagicNumber() {
    this->mMagicNumber = 911;
}


template <typename T>
void
DataType::ConvertPrecisionDispatcher(const Precision &aPrecision) {

    auto data = (T *) this->mpData;
    auto size = this->mSize;
    this->mPrecision = aPrecision;


    if (size == 0) {
        return;
    }
    switch (aPrecision) {
        case HALF: {
            auto temp = new float16[size];
            std::copy(data, data + size, temp);
            this->SetData((char *) temp);
            break;
        }
        case FLOAT: {
            auto temp = new float[size];
            std::copy(data, data + size, temp);
            this->SetData((char *) temp);
            break;
        }
        case DOUBLE: {
            auto temp = new double[size];
            std::copy(data, data + size, temp);
            this->SetData((char *) temp);
            break;
        }
        default: {
            MPCR_API_EXCEPTION("Invalid Precision : Not Supported", -1);
        }
    }

}


void
DataType::ConvertPrecision(const mpcr::precision::Precision &aPrecision) {
    if (mPrecision == aPrecision) {
        return;
    }
    SIMPLE_DISPATCH(this->mPrecision, ConvertPrecisionDispatcher, aPrecision)
}


template <typename T>
void
DataType::ConvertToVector(std::vector <double> &aOutput) {
    auto pData = (T *) this->mpData;
    aOutput.clear();
    aOutput.resize(this->mSize);
    aOutput.assign(pData, pData + this->mSize);
}


std::vector <double> *
DataType::ConvertToNumericVector() {
    auto pOutput = new std::vector <double>();
    SIMPLE_DISPATCH(this->mPrecision, ConvertToVector, *pOutput)
    return pOutput;
}


Rcpp::NumericMatrix *
DataType::ConvertToRMatrix() {
    if (!this->mMatrix) {
        MPCR_API_EXCEPTION("Invalid Cannot Convert, Not a Matrix", -1);
    }
    Rcpp::NumericMatrix *pOutput = nullptr;

    SIMPLE_DISPATCH(this->mPrecision, ConvertToRMatrixDispatcher, pOutput)
    return pOutput;

}


template <typename T>
void DataType::ConvertToRMatrixDispatcher(Rcpp::NumericMatrix *&aOutput) {

    auto pData = (T *) mpData;
    aOutput = new Rcpp::NumericMatrix(this->mpDimensions->GetNRow(),
                                      this->mpDimensions->GetNCol(), pData);

}


template <typename T>
void DataType::CheckNA(std::vector <int> &aOutput, Dimensions *&apDimensions) {
    auto pData = (T *) this->mpData;
    aOutput.clear();
    aOutput.resize(this->mSize);
    if (this->mMatrix) {
        delete apDimensions;
        apDimensions = new Dimensions(this->mpDimensions->GetNRow(),
                                      this->mpDimensions->GetNCol());

    }

    for (auto i = 0; i < this->mSize; i++) {
        aOutput[ i ] = std::isnan(pData[ i ]);
    }

}


std::vector <int> *
DataType::IsNA(Dimensions *&apDimensions) {
    auto pOutput = new std::vector <int>();
    SIMPLE_DISPATCH(this->mPrecision, CheckNA, *pOutput, apDimensions)
    return pOutput;
}


DataType *
DataType::PerformPlusDispatcher(SEXP aObj) {

    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformPlus(this, val, "");

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return RPerformPlus(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RPerformPlus(this, temp_mpr);
    }
}


DataType *
DataType::PerformPowDispatcher(SEXP aObj) {

    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformPow(this, val, "");

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return RPerformPow(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RPerformPow(this, temp_mpr);
    }
}


DataType *
DataType::PerformDivDispatcher(SEXP aObj) {
    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformDiv(this, val, "");

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return RPerformDiv(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RPerformDiv(this, temp_mpr);
    }
}


DataType *
DataType::PerformMultDispatcher(SEXP aObj) {

    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformMult(this, val, "");

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return RPerformMult(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RPerformMult(this, temp_mpr);
    }
}


DataType *
DataType::PerformMinusDispatcher(SEXP aObj) {

    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformMinus(this, val, "");

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return RPerformMinus(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RPerformMinus(this, temp_mpr);
    }
}


SEXP
DataType::GreaterThanDispatcher(SEXP aObj) {

    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RGreaterThan(this, val);

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return RGreaterThan(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RGreaterThan(this, temp_mpr);
    }
}


SEXP
DataType::GreaterThanOrEqualDispatcher(SEXP aObj) {
    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RGreaterThanOrEqual(this, val);

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return RGreaterThanOrEqual(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RGreaterThanOrEqual(this, temp_mpr);
    }
}


SEXP
DataType::LessThanDispatcher(SEXP aObj) {
    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RLessThan(this, val);

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return RLessThan(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RLessThan(this, temp_mpr);
    }
}


SEXP
DataType::LessThanOrEqualDispatcher(SEXP aObj) {
    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RLessThanOrEqual(this, val);

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return RLessThanOrEqual(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RLessThanOrEqual(this, temp_mpr);
    }
}


SEXP
DataType::EqualDispatcher(SEXP aObj) {
    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return REqual(this, val);

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return REqual(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return REqual(this, temp_mpr);
    }
}


SEXP
DataType::NotEqualDispatcher(SEXP aObj) {
    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RNotEqual(this, val);

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        auto temp_mpr = new DataType(0, DOUBLE);
        temp_mpr->SetSize(values.size());
        temp_mpr->SetData((char *) values.data());
        return RNotEqual(this, temp_mpr);

    } else {
        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
            aObj);
        if (!temp_mpr->IsDataType()) {
            MPCR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MPR Object",
                -1);
        }
        return RNotEqual(this, temp_mpr);
    }
}


void
DataType::Transpose() {
    if (!this->mMatrix) {
        MPCR_API_EXCEPTION("Cannot Transpose a Vector", -1);
    }
    SIMPLE_DISPATCH(this->mPrecision, DataType::TransposeDispatcher)
}


template <typename T>
void
DataType::TransposeDispatcher() {

    auto pData = (T *) this->mpData;
    auto pOutput = new T[this->mSize];
    auto col = this->GetNCol();
    auto row = this->GetNRow();

    size_t counter = 0;
    size_t idx;

    for (auto i = 0; i < row; i++) {
        for (auto j = 0; j < col; j++) {
            idx = ( j * row ) + i;
            pOutput[ counter ] = pData[ idx ];
            counter++;
        }
    }

    this->SetData((char *) pOutput);
    this->SetDimensions(col, row);

}


void DataType::SetValues(std::vector <double> &aValues) {
    this->mSize = aValues.size();
    if (this->mMatrix) {
        delete this->mpDimensions;
        this->mpDimensions = nullptr;
        this->mMatrix = false;
    }
    delete[] this->mpData;
    this->mpData = nullptr;

    SIMPLE_DISPATCH(this->mPrecision, Init, &aValues)
}


void DataType::FillTriangle(const double &aValue, const bool &aUpperTriangle) {
    SIMPLE_DISPATCH(this->mPrecision, DataType::FillTriangleDispatcher, aValue,
                    aUpperTriangle)
}


template <typename T>
void DataType::FillTriangleDispatcher(const double &aValue,
                                      const bool &aUpperTriangle) {

    auto row = this->GetNRow();
    auto col = this->GetNCol();
    auto pData = (T *) this->mpData;

    if (!aUpperTriangle) {
        for (auto j = 0; j < col; j++) {
            for (auto i = j + 1; i < row; i++)
                pData[ i + row * j ] = aValue;
        }
    } else {
        for (auto i = 0; i < row; i++) {
            for (auto j = i + 1; j < col; j++) {
                pData[ i + row * j ] = aValue;
            }
        }
    }

}


double
DataType::Sum() {
    double sum;
    SIMPLE_DISPATCH(this->mPrecision, DataType::SumDispatcher, sum)
    return sum;
}


template <typename T>
void
DataType::SumDispatcher(double &aResult) {
    aResult = 0;
    auto pData = (T *) this->mpData;
    for (auto i = 0; i < this->mSize; i++) {
        aResult += pData[ i ];
    }
}


double
DataType::Product() {
    double prod;
    SIMPLE_DISPATCH(this->mPrecision, DataType::ProductDispatcher, prod)
    return prod;
}


template <typename T>
void
DataType::ProductDispatcher(double &aResult) {
    aResult = 1;
    auto pData = (T *) this->mpData;
    for (auto i = 0; i < this->mSize; i++) {
        aResult *= pData[ i ];
    }
}


double DataType::Determinant() {
    if (!this->mMatrix) {
        MPCR_API_EXCEPTION("Cannot calculate determinant for a vector", -1);
    }
    if (this->GetNRow() != this->GetNCol()) {
        MPCR_API_EXCEPTION(
            "Cannot calculate determinant for a non-square matrix", -1);
    }
    double result;
    SIMPLE_DISPATCH(this->mPrecision, DataType::DeterminantDispatcher, result)
    return result;
}


template <typename T>
void
DataType::DeterminantDispatcher(double &aResult) {

    double det = 1.0;
    auto data = (T *) this->mpData;
    auto size = this->GetNCol();
    std::vector <double> pData;

    if (size == 2) {
        aResult = data[ 0 ] * data[ 3 ] - data[ 1 ] * data[ 2 ];
        return;
    }

    pData.resize(this->mSize);
    std::copy(data, data + this->mSize, pData.begin());


    for (int i = 0; i < size; i++) {
        int max_row = i;
        for (int j = i + 1; j < size; j++) {
            if (abs(pData[ j * size + i ]) >
                abs(pData[ max_row * size + i ])) {
                max_row = j;
            }
        }
        if (max_row != i) {
            swap_ranges(pData.begin() + i * size,
                        pData.begin() + ( i + 1 ) * size,
                        pData.begin() + max_row * size);
            det = -det;
        }
        det *= pData[ i * size + i ];
        if (pData[ i * size + i ] == 0) {
            aResult = 0;
            return;
        }
        for (int j = i + 1; j < size; j++) {
            double factor = pData[ j * size + i ] / pData[ i * size + i ];
            for (int k = i + 1; k < size; k++) {
                pData[ j * size + k ] -= factor * pData[ i * size + k ];
            }
        }
    }
    aResult = det;
}


std::vector <char>
DataType::Serialize() {
    size_t size = 1;
    auto size_val = 0;
    auto itr = 0;
    char metadata = 0;

    if (this->mPrecision == mpcr::precision::FLOAT) {
        size_val += sizeof(float);

    } else if (this->mPrecision == mpcr::precision::DOUBLE) {
        size_val += sizeof(double);
    }

    size += this->mSize * size_val;

    if (this->mMatrix) {
        size += sizeof(size_t) * 2;
        metadata |= 0x80;
    } else {
        size += sizeof(size_t);
    }

    metadata |= (( static_cast<int>(this->mPrecision ) & 0x03 ) << 5 );

    std::vector <char> vec;
    vec.resize(size);

    auto buffer = vec.data();
    buffer[ 0 ] = metadata;

    if (this->mMatrix) {
        memcpy(buffer + 1, (char *) &this->mpDimensions->mRow, sizeof(size_t));
        memcpy(buffer + 1 + sizeof(size_t), (char *) &this->mpDimensions->mCol,
               sizeof(size_t));

        itr = 1 + ( sizeof(size_t) * 2 );
    } else {
        memcpy(buffer + 1, (char *) &this->mSize, sizeof(size_t));
        itr = 1 + sizeof(size_t);
    }

    memcpy(buffer + itr, this->mpData, this->mSize * size_val);

    return vec;
}


DataType *
DataType::DeSerialize(char *apData) {
    auto metadata = apData[ 0 ];
    bool is_matrix = (( metadata & 0x80 ) != 0 );
    auto temp_precision = static_cast<Precision>((( metadata >> 5 ) & 0x03 ));

    auto itr = 0;

    auto ret = new DataType(temp_precision);
    ret->ClearUp();

    auto obj_size = sizeof(float);
    if (temp_precision == DOUBLE) {
        obj_size = sizeof(double);
    }

    if (is_matrix) {
        size_t row = *(size_t *) ( apData + 1 );
        size_t col = *((size_t *) ( apData + 1 ) + 1 );
        ret->SetSize(row * col);
        ret->SetDimensions(row, col);
        itr = 1 + ( sizeof(size_t) * 2 );
    } else {
        size_t size = *(size_t *) ( apData + 1 );
        ret->SetSize(size);
        itr = 1 + sizeof(size_t);
    }

    auto temp_data = new char[ret->GetSize() * obj_size];
    memcpy(temp_data, apData + itr, obj_size * ret->GetSize());
    ret->SetData(temp_data);

    return ret;
}


Rcpp::RawVector
DataType::RSerialize() {
    size_t size = 1;
    auto size_val = 0;
    auto itr = 0;
    char metadata = 0;

    if (this->mPrecision == mpcr::precision::FLOAT) {
        size_val += sizeof(float);

    } else if (this->mPrecision == mpcr::precision::DOUBLE) {
        size_val += sizeof(double);
    }

    size += this->mSize * size_val;

    if (this->mMatrix) {
        size += sizeof(size_t) * 2;
        metadata |= 0x80;
    } else {
        size += sizeof(size_t);
    }

    metadata |= (( static_cast<int>(this->mPrecision ) & 0x03 ) << 5 );

    Rcpp::RawVector vec(size);

    auto buffer = vec.begin();
    vec[ 0 ] = metadata;

    if (this->mMatrix) {
        memcpy(buffer + 1, (char *) &this->mpDimensions->mRow, sizeof(size_t));
        memcpy(buffer + 1 + sizeof(size_t), (char *) &this->mpDimensions->mCol,
               sizeof(size_t));

        itr = 1 + ( sizeof(size_t) * 2 );
    } else {
        memcpy(buffer + 1, (char *) &this->mSize, sizeof(size_t));
        itr = 1 + sizeof(size_t);
    }

    memcpy(buffer + itr, this->mpData, this->mSize * size_val);

    return vec;
}


DataType *
DataType::RDeSerialize(Rcpp::RawVector aInput) {
    auto metadata = aInput[ 0 ];
    bool is_matrix = (( metadata & 0x80 ) != 0 );
    auto temp_precision = static_cast<Precision>((( metadata >> 5 ) & 0x03 ));

    auto itr = 0;

    auto ret = new DataType(temp_precision);
    ret->ClearUp();

    auto obj_size = sizeof(float);
    if (temp_precision == DOUBLE) {
        obj_size = sizeof(double);
    }

    auto data = aInput.begin();

    if (is_matrix) {
        size_t row = *(size_t *) ( data + 1 );
        size_t col = *((size_t *) ( data + 1 ) + 1 );
        ret->SetSize(row * col);
        ret->SetDimensions(row, col);
        itr = 1 + ( sizeof(size_t) * 2 );
    } else {
        size_t size = *(size_t *) ( data + 1 );
        ret->SetSize(size);
        itr = 1 + sizeof(size_t);
    }

    auto temp_data = new char[ret->GetSize() * obj_size];
    memcpy(temp_data, data + itr, obj_size * ret->GetSize());
    ret->SetData(temp_data);

    return ret;
}


SIMPLE_INSTANTIATE(void, DataType::DeterminantDispatcher, double &aResult)

SIMPLE_INSTANTIATE(void, DataType::ProductDispatcher, double &aResult)

SIMPLE_INSTANTIATE(void, DataType::SumDispatcher, double &aResult)

SIMPLE_INSTANTIATE(void, DataType::FillTriangleDispatcher, const double &aValue,
                   const bool &aUpperTriangle)

SIMPLE_INSTANTIATE(void, DataType::CheckNA, std::vector <int> &aOutput,
                   Dimensions *&apDimensions)

SIMPLE_INSTANTIATE(void, DataType::ConvertPrecisionDispatcher,
                   const Precision &aPrecision)

SIMPLE_INSTANTIATE(void, DataType::CheckNA, const size_t &aIndex, bool &aFlag)

SIMPLE_INSTANTIATE(void, DataType::Init, std::vector <double> *aValues)

SIMPLE_INSTANTIATE(void, DataType::PrintVal)

SIMPLE_INSTANTIATE(void, DataType::GetCopyOfData, const char *apSrc,
                   char *&apDest)

SIMPLE_INSTANTIATE(void, DataType::GetValue, size_t aIndex, double &aOutput)

SIMPLE_INSTANTIATE(void, DataType::SetValue, size_t aIndex, double &aVal)

SIMPLE_INSTANTIATE(void, DataType::GetDataSize, size_t &aDataSize)

SIMPLE_INSTANTIATE(void, DataType::ConvertToVector,
                   std::vector <double> &aOutput)

SIMPLE_INSTANTIATE(void, DataType::ConvertToRMatrixDispatcher,
                   Rcpp::NumericMatrix *&aOutput)

SIMPLE_INSTANTIATE(void, DataType::TransposeDispatcher)

SIMPLE_INSTANTIATE(void, DataType::PrintRowsDispatcher, const size_t &aRowIdx,
                   std::stringstream &aRowAsString)

INSTANTIATE(void, DataType::GetCopyOfData, DataType &aSrc,
            DataType &aDestination)