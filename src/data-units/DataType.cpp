
#include <data-units/DataType.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <adapters/RBinaryOperations.hpp>
#include <Rcpp.h>


using namespace mpr::precision;


DataType::DataType(size_t aSize, Precision aPrecision) {
    this->SetMagicNumber();
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mSize = aSize;
    this->mpDimensions = nullptr;
    this->mMatrix = false;
    SIMPLE_DISPATCH(this->mPrecision, Init)
}


DataType::DataType(size_t aSize, int aPrecision) {
    this->SetMagicNumber();
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mpDimensions = nullptr;
    this->mMatrix = false;
    this->mSize = aSize;
    SIMPLE_DISPATCH(this->mPrecision, Init)
}


DataType::DataType(size_t aSize, const std::string &aPrecision) {
    this->SetMagicNumber();
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mpDimensions = nullptr;
    this->mMatrix = false;
    this->mSize = aSize;
    SIMPLE_DISPATCH(this->mPrecision, Init)

}


DataType::DataType(size_t aRow, size_t aCol, Precision aPrecision) {
    this->SetMagicNumber();
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mpDimensions = new Dimensions(aRow, aCol);
    this->mMatrix = true;
    this->mSize = aRow * aCol;
    SIMPLE_DISPATCH(this->mPrecision, Init)
}


DataType::DataType(mpr::precision::Precision aPrecision) {
    this->SetMagicNumber();
    this->mPrecision = GetInputPrecision(aPrecision);
    this->mMatrix = false;
    this->mpDimensions = nullptr;
    this->mSize = 0;
    this->mpData = nullptr;
}


DataType::DataType(DataType &aDataType) {
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


DataType::~DataType() {
    delete[] mpData;
    delete mpDimensions;
}


template <typename T>
void
DataType::Init() {
    T *temp = new T[mSize];
    for (auto i = 0; i < mSize; i++) {
        temp[ i ] = (T) 1.5;
    }
    this->mpData = (char *) temp;

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
        ss << "Number of Rows : " << rows << std::endl;
        ss << "Number of Columns : " << cols << std::endl;
        ss << "---------------------" << std::endl;
        size_t start_idx;
        size_t print_col = ( cols > 13 ) ? 13 : cols;
        size_t print_rows = ( rows > 100 ) ? 100 : rows;

        for (auto i = 0; i < print_rows; i++) {
            start_idx = i * cols;
            ss << " [\t";
            for (auto j = 0; j < print_col; j++) {
                ss << temp[ start_idx + j ] << "\t";
            }
            ss << "]" << std::endl;
            if (ss.gcount() > stream_size) {
                Rcpp::Rcout << std::string(ss.str());
                ss.clear();
            }
        }
        if (print_rows * print_col != this->mSize) {
            ss << "Note Only Matrix with size 100*13 is printed" << std::endl;
        }
        Rcpp::Rcout << std::string(ss.str());


    } else {
        ss << "Vector Size : " << mSize << std::endl;
        ss << "---------------------" << std::endl;
        ss << " [\t";
        for (auto i = 0; i < mSize; i++) {
            ss << temp[ i ] << "\t";
            if (i % 100 == 0) {
                if (ss.gcount() > stream_size) {
                    Rcpp::Rcout << std::string(ss.str());
                    ss.clear();
                }
            }
        }
        ss << "]" << std::endl;
        Rcpp::Rcout << std::string(ss.str());
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
DataType::GetValue(size_t aIndex, double *&aOutput) {
    double *temp = new double;
    *temp = (double) (((T *) this->mpData )[ aIndex ] );
    delete aOutput;
    aOutput = temp;
}


double
DataType::GetVal(size_t aIndex) {
    if (aIndex >= this->mSize) {
        MPR_API_EXCEPTION("Segmentation Fault Index Out Of Bound", -1);
    }
    double *temp = nullptr;
    SIMPLE_DISPATCH(mPrecision, GetValue, aIndex, temp)
    return *temp;
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
        MPR_API_EXCEPTION("Segmentation Fault Index Out Of Bound", -1);
    }
    SIMPLE_DISPATCH(mPrecision, SetValue, aIndex, aVal)

}


void
DataType::SetPrecision(mpr::precision::Precision aPrecision) {
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
        MPR_API_EXCEPTION("Not a Matrix Fault.", -1);
    }
    if (aRow > mpDimensions->GetNRow() || aCol > mpDimensions->GetNCol() ||
        aRow < 0 || aCol < 0) {
        MPR_API_EXCEPTION("Segmentation Fault Index Out Of Bound", -1);
    }
    return ( aRow * mpDimensions->GetNCol()) + aCol;
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
        MPR_API_EXCEPTION("Segmentation Fault Matrix Out Of Bound", -1);
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


DataType *
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
    return this;
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

    if (size == 0) {
        return;
    }
    switch (aPrecision) {
        case INT: {
            auto temp = new int[size];
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
            MPR_API_EXCEPTION("Invalid Precision : Not Supported", -1);
        }
    }

    this->mPrecision = aPrecision;
}


void
DataType::ConvertPrecision(const mpr::precision::Precision &aPrecision) {
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
        MPR_API_EXCEPTION("Invalid Cannot Convert, Not a Matrix", -1);
    }

    auto pOutput = new Rcpp::NumericMatrix(this->mpDimensions->GetNRow(),
                                           this->mpDimensions->GetNCol());

    SIMPLE_DISPATCH(this->mPrecision, ConvertToRMatrixDispatcher, *pOutput)
    return pOutput;

}


template <typename T>
void DataType::ConvertToRMatrixDispatcher(Rcpp::NumericMatrix &aOutput) {

    auto pData = (T *) this->mpData;
    size_t itr_vec = 0;

    for (auto i = 0; i < this->mpDimensions->GetNRow(); i++) {
        auto row = aOutput.row(i);
        for (auto itr = row.begin(); itr != row.end(); itr++) {
            *itr = pData[ itr_vec ];
            itr_vec++;
        }
    }

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
DataType::operator +(const DataType &aInput) {

    Rcpp::Rcout<<"hereeeeeeeeeeeee"<<std::endl;
    //    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RPerformPlus(this, val, "");
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
////        }
//        DataType temp=aInput;
//        return RPerformPlus(this, aInput );
//    }
        return new DataType(50,FLOAT);
}
//
//DataType *
//DataType::operator ^(SEXP aInput) {
//
//    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RPerformPow(this, val, "");
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
//        }
//        return RPerformPow(this, temp_mpr);
//    }
//}
//
//
//DataType *
//DataType::operator /(SEXP aInput) {
//    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RPerformDiv(this, val, "");
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
//        }
//        return RPerformDiv(this, temp_mpr);
//    }
//}
//
//
//DataType *
//DataType::operator *(SEXP aInput) {
//
//    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RPerformMult(this, val, "");
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
//        }
//        return RPerformMult(this, temp_mpr);
//    }
//}
//
//
//DataType *
//DataType::operator -(SEXP aInput) {
//
//    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RPerformMinus(this, val, "");
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
//        }
//        return RPerformMinus(this, temp_mpr);
//    }
//}
//
//
//SEXP
//DataType::operator >(SEXP aInput) {
//
//    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RGreaterThan(this, val);
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
//        }
//        return RGreaterThan(this, temp_mpr);
//    }
//}
//
//
//SEXP
//DataType::operator >=(SEXP aInput) {
//    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RGreaterThanOrEqual(this, val);
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
//        }
//        return RGreaterThanOrEqual(this, temp_mpr);
//    }
//}
//
//
//SEXP
//DataType::operator <(SEXP aInput) {
//    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RLessThan(this, val);
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
//        }
//        return RLessThan(this, temp_mpr);
//    }
//}
//
//
//SEXP
//DataType::operator <=(SEXP aInput) {
//    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RLessThanOrEqual(this, val);
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
//        }
//        return RLessThanOrEqual(this, temp_mpr);
//    }
//}
//
//
//SEXP
//DataType::operator ==(SEXP aInput) {
//    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return REqual(this, val);
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
//        }
//        return REqual(this, temp_mpr);
//    }
//}
//
//
//SEXP
//DataType::operator !=(SEXP aInput) {
//    if (TYPEOF(aInput) == REALSXP || TYPEOF(aInput) == INTSXP) {
//        auto val = Rcpp::as <double>(aInput);
//        return RNotEqual(this, val);
//
//    } else {
//        auto temp_mpr = (DataType *) Rcpp::internal::as_module_object_internal(
//            aInput);
//        if (!temp_mpr->IsDataType()) {
//            MPR_API_EXCEPTION(
//                "Undefined Object . Make Sure You're Using MPR Object",
//                -1);
//        }
//        return RNotEqual(this, temp_mpr);
//    }
//}


SIMPLE_INSTANTIATE(void, DataType::CheckNA, std::vector <int> &aOutput,
                   Dimensions *&apDimensions)

SIMPLE_INSTANTIATE(void, DataType::ConvertPrecisionDispatcher,
                   const Precision &aPrecision)

SIMPLE_INSTANTIATE(void, DataType::CheckNA, const size_t &aIndex, bool &aFlag)

SIMPLE_INSTANTIATE(void, DataType::Init)

SIMPLE_INSTANTIATE(void, DataType::PrintVal)

SIMPLE_INSTANTIATE(void, DataType::GetCopyOfData, const char *apSrc,
                   char *&apDest)

SIMPLE_INSTANTIATE(void, DataType::GetValue, size_t aIndex, double *&aOutput)

SIMPLE_INSTANTIATE(void, DataType::SetValue, size_t aIndex, double &aVal)

SIMPLE_INSTANTIATE(void, DataType::GetDataSize, size_t &aDataSize)

SIMPLE_INSTANTIATE(void, DataType::ConvertToVector,
                   std::vector <double> &aOutput)

SIMPLE_INSTANTIATE(void, DataType::ConvertToRMatrixDispatcher,
                   Rcpp::NumericMatrix &aOutput)