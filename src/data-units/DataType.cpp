/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <data-units/DataType.hpp>
#include <adapters/RBinaryOperations.hpp>


using namespace mpcr::precision;


/** ------------------------- Constructors ---------------------------------- **/

void DataType::InitializeObject(size_t aSize, const Precision &aPrecision,
                                const OperationPlacement &aOperationPlacement) {
    this->SetPrecision(aPrecision, aOperationPlacement);
    this->mpDimensions = nullptr;
    this->SetMagicNumber();
    mData.ClearUp();
    this->mSize = aSize;
    this->mMatrix = false;
}


DataType::DataType(size_t aSize, Precision aPrecision,
                   const OperationPlacement &aOperationPlacement) {

    this->InitializeObject(aSize, aPrecision, aOperationPlacement);
    SIMPLE_DISPATCH_WITH_HALF(this->mPrecision, Init, nullptr,
                              aOperationPlacement)
}


DataType::DataType(std::vector <double> aValues, std::string aPrecision,
                   const OperationPlacement &aOperationPlacement) {

    auto precision = GetInputPrecision(aPrecision);
    this->InitializeObject(aValues.size(), precision, aOperationPlacement);

    SIMPLE_DISPATCH_WITH_HALF(this->mPrecision, Init, &aValues,
                              aOperationPlacement)

}


DataType::DataType(std::vector <double> &aValues, const size_t &aRow,
                   const size_t &aCol, const std::string &aPrecision,
                   const OperationPlacement &aOperationPlacement) {

    auto precision = GetInputPrecision(aPrecision);
    this->InitializeObject(aValues.size(), precision, aOperationPlacement);

    this->mpDimensions = new Dimensions(aRow, aCol);
    this->mMatrix = true;

    SIMPLE_DISPATCH_WITH_HALF(this->mPrecision, Init, &aValues,
                              aOperationPlacement)
}


DataType::DataType(std::vector <double> &aValues,
                   mpcr::definitions::Precision aPrecision,
                   const OperationPlacement &aOperationPlacement) {
    auto precision = GetInputPrecision(aPrecision);
    this->InitializeObject(aValues.size(), precision, aOperationPlacement);
    SIMPLE_DISPATCH_WITH_HALF(this->mPrecision, Init, &aValues,
                              aOperationPlacement)
}


DataType::DataType(size_t aSize, int aPrecision,
                   const OperationPlacement &aOperationPlacement) {
    auto precision = GetInputPrecision(aPrecision);
    this->InitializeObject(aSize, precision, aOperationPlacement);

    SIMPLE_DISPATCH_WITH_HALF(this->mPrecision, Init, nullptr,
                              aOperationPlacement)
}


DataType::DataType(size_t aSize, const std::string &aPrecision,
                   const OperationPlacement &aOperationPlacement) {

    auto precision = GetInputPrecision(aPrecision);
    this->InitializeObject(aSize, precision, aOperationPlacement);
    SIMPLE_DISPATCH_WITH_HALF(this->mPrecision, Init, nullptr,
                              aOperationPlacement)

}


DataType::DataType(size_t aRow, size_t aCol, Precision aPrecision,
                   const OperationPlacement &aOperationPlacement) {

    auto precision = GetInputPrecision(aPrecision);
    this->InitializeObject(aRow * aCol, precision, aOperationPlacement);

    this->mpDimensions = new Dimensions(aRow, aCol);
    this->mMatrix = true;

    SIMPLE_DISPATCH_WITH_HALF(this->mPrecision, Init)
}


DataType::DataType(mpcr::definitions::Precision aPrecision,
                   const OperationPlacement &aOperationPlacement) {
    this->InitializeObject(0, aPrecision, aOperationPlacement);
}


DataType::DataType(const DataType &aDataType) {
    this->SetMagicNumber();
    this->ClearUp();
    this->mSize = aDataType.mSize;
    this->mPrecision = aDataType.mPrecision;
    this->mMatrix = aDataType.mMatrix;
    this->mData = aDataType.mData;

    if (this->mMatrix) {
        this->mpDimensions = new Dimensions(*aDataType.GetDimensions());
    }
}


DataType::DataType(DataType &aDataType,
                   const mpcr::definitions::Precision &aPrecision) {
    this->SetMagicNumber();
    this->ClearUp();
    this->mSize = aDataType.mSize;
    this->mPrecision = aPrecision;
    this->mMatrix = aDataType.mMatrix;
    this->mData = aDataType.mData;
    if (this->mMatrix) {
        this->mpDimensions = new Dimensions(*aDataType.GetDimensions());
    }
    SIMPLE_DISPATCH_WITH_HALF(aDataType.mPrecision, ConvertPrecisionDispatcher,
                              this->mPrecision)
}


DataType::~DataType() {
    delete mpDimensions;
}


/** ---------------------------- Methods ------------------------------- **/



std::string
DataType::PrintRow(const size_t &aRowIdx) {

    if (aRowIdx > this->GetNRow()) {
        MPCR_API_EXCEPTION("Segmentation fault index out of Bound", -1);
    }

    this->CheckHalfCompatibility();

    std::stringstream ss;
    SIMPLE_DISPATCH(this->mPrecision, DataType::PrintRowsDispatcher, aRowIdx,
                    ss)
    return ss.str();

}


void
DataType::Print() {
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(mPrecision, PrintVal)
}


Precision &
DataType::GetPrecision() {
    return this->mPrecision;
}


char *
DataType::GetData(const OperationPlacement &aOperationPlacement) {
    this->CheckHalfCompatibility(aOperationPlacement);
    return mData.GetDataPointer(aOperationPlacement);
}


void
DataType::Allocate(std::vector <double> &aValues,
                   const OperationPlacement &aPlacement) {

    this->SetPrecision(this->mPrecision, aPlacement);
    this->mSize = aValues.size();
    SIMPLE_DISPATCH_WITH_HALF(this->mPrecision, Init, &aValues, aPlacement)
}


size_t
DataType::GetSize() const {
    return this->mSize;
}


double
DataType::GetVal(size_t aIndex) {
    double temp = 0;
    if (aIndex >= this->mSize) {
        MPCR_API_EXCEPTION("Segmentation Fault Index Out Of Bound", -1);
    }
    this->CheckHalfCompatibility();

    SIMPLE_DISPATCH(mPrecision, GetValue, aIndex, temp)
    return temp;
}


void
DataType::SetVal(size_t aIndex, double aVal) {
    if (aIndex >= this->mSize) {
        MPCR_API_EXCEPTION("Segmentation Fault Index Out Of Bound", -1);
    }
    this->CheckHalfCompatibility();

    SIMPLE_DISPATCH(mPrecision, SetValue, aIndex, aVal)

}


void
DataType::SetPrecision(mpcr::definitions::Precision aPrecision,
                       const OperationPlacement &aOperationPlacement) {
    this->ClearUp();
    if (aPrecision == HALF && aOperationPlacement == CPU) {
        this->mPrecision = FLOAT;
        MPCR_PRINTER("Cannot allocate 16-bit precision on CPU, ")
        MPCR_PRINTER("Changed to 32-bit")
        MPCR_PRINTER(std::endl)
    } else {
        this->mPrecision = aPrecision;
    }
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
DataType::SetData(char *aData, const OperationPlacement &aOperationPlacement) {
    auto op_placement = aOperationPlacement;
    if (this->mPrecision == HALF && aOperationPlacement == CPU) {
        MPCR_API_EXCEPTION("Cannot allocate 16-bit precision on CPU", -1);
    }
    this->mData.SetDataPointer(aData, this->GetSizeInBytes(),
                               op_placement);
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


DataType &
DataType::operator =(const DataType &aDataType) {
    this->mSize = aDataType.mSize;
    this->mPrecision = aDataType.mPrecision;
    this->mMatrix = aDataType.mMatrix;
    mData = aDataType.mData;
    if (this->mMatrix) {
        this->mpDimensions = new Dimensions(*aDataType.GetDimensions());
    } else {
        this->mpDimensions = nullptr;
    }

    return *this;
}


bool
DataType::IsNA(const size_t &aIndex) {
    bool flag = false;
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(this->mPrecision, CheckNA, aIndex, flag)
    return flag;
}


size_t
DataType::GetObjectSize() {
    auto size = this->GetSizeInBytes();
    int num = 0;

    if (IsCPUAllocated()) {
        num++;
    }
    if (IsGPUAllocated()) {
        num++;
    }

    size_t data_size = size * num;
    if (this->mMatrix) {
        data_size += 3 * sizeof(size_t);
    } else {
        data_size += sizeof(size_t);
    }
    data_size += sizeof(bool);
    data_size += sizeof(Precision);
    data_size += sizeof(DataHolder);
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


void
DataType::SetMagicNumber() {
    this->mMagicNumber = 911;
}


void
DataType::ConvertPrecision(const mpcr::definitions::Precision &aPrecision) {
    auto temp_precision = aPrecision;
    if (mPrecision == aPrecision) {
        return;
    }

#ifndef USE_CUDA
        if(aPrecision==HALF){
        temp_precision=FLOAT;
        MPCR_PRINTER("Cannot allocate 16-bit precision with CPU only compiled code. ")
        MPCR_PRINTER("Changing to 32-bit precision")
        MPCR_PRINTER(std::endl)
    }
#endif

    SIMPLE_DISPATCH_WITH_HALF(this->mPrecision, ConvertPrecisionDispatcher,
                              temp_precision)
}


std::vector <double> *
DataType::ConvertToNumericVector() {
    auto pOutput = new std::vector <double>();
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(this->mPrecision, ConvertToVector, *pOutput)
    return pOutput;
}


Rcpp::NumericMatrix *
DataType::ConvertToRMatrix() {
    if (!this->mMatrix) {
        MPCR_API_EXCEPTION("Invalid Cannot Convert, Not a Matrix", -1);
    }
    Rcpp::NumericMatrix *pOutput = nullptr;
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(this->mPrecision, ConvertToRMatrixDispatcher, pOutput)
    return pOutput;

}


std::vector <int> *
DataType::IsNA(Dimensions *&apDimensions) {
    auto pOutput = new std::vector <int>();
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(this->mPrecision, CheckNA, *pOutput, apDimensions)
    return pOutput;
}


void
DataType::Transpose() {
    if (!this->mMatrix) {
        MPCR_API_EXCEPTION("Cannot Transpose a Vector", -1);
    }
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(this->mPrecision, DataType::TransposeDispatcher)
}


void
DataType::FillTriangle(const double &aValue, const bool &aUpperTriangle) {
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(this->mPrecision, DataType::FillTriangleDispatcher, aValue,
                    aUpperTriangle)
}


double
DataType::Sum() {
    double sum;
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(this->mPrecision, DataType::SumDispatcher, sum)
    return sum;
}


double
DataType::SquareSum() {
    double sum;
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(this->mPrecision, DataType::SquareSumDispatcher, sum)
    return sum;

}


double
DataType::Product() {
    double prod;
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(this->mPrecision, DataType::ProductDispatcher, prod)
    return prod;
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
    this->CheckHalfCompatibility();
    SIMPLE_DISPATCH(this->mPrecision, DataType::DeterminantDispatcher, result)
    return result;
}


std::vector <char>
DataType::Serialize() {
    this->CheckHalfCompatibility();

    size_t size = 1;
    auto size_val = 0;
    auto itr = 0;
    char metadata = 0;

    if (this->mPrecision == mpcr::definitions::FLOAT) {
        size_val += sizeof(float);

    } else if (this->mPrecision == mpcr::definitions::DOUBLE) {
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

    memcpy(buffer + itr, this->GetData(CPU), this->mSize * size_val);

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

    auto temp_data = mpcr::memory::AllocateArray(ret->GetSize() * obj_size, CPU,
                                                 nullptr);
    memcpy(temp_data, apData + itr, obj_size * ret->GetSize());
    ret->SetData(temp_data);

    return ret;
}


Rcpp::RawVector
DataType::RSerialize() {
    this->CheckHalfCompatibility();

    size_t size = 1;
    auto size_val = 0;
    auto itr = 0;
    char metadata = 0;

    auto pData = this->GetData(CPU);

    if (this->mPrecision == mpcr::definitions::FLOAT) {
        size_val += sizeof(float);

    } else if (this->mPrecision == mpcr::definitions::DOUBLE) {
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

    memcpy(buffer + itr, pData, this->mSize * size_val);

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


    auto temp_data = mpcr::memory::AllocateArray(ret->GetSize() * obj_size, CPU,
                                                 nullptr);
    memcpy(temp_data, data + itr, obj_size * ret->GetSize());
    ret->SetData(temp_data);

    return ret;
}


size_t
DataType::GetSizeInBytes() {
    size_t size = this->mSize;
    switch (this->mPrecision) {
        case HALF:
            return size * sizeof(float16);
        case FLOAT:
            return size * sizeof(float);
        case DOUBLE:
            return size * sizeof(double);
        default:
            MPCR_API_EXCEPTION("Error while getting size in bytes", -1);
    }
    return 0;
}


void
DataType::PrintTotalSize() {

    auto size = this->GetSizeInBytes();

    if (IsCPUAllocated()) {
        MPCR_PRINTER("Total memory allocated on CPU in bytes : ")
        MPCR_PRINTER(size)
        MPCR_PRINTER(std::endl)
    } else {
        MPCR_PRINTER("No allocations on CPU")
        MPCR_PRINTER(std::endl)
    }

    if (IsGPUAllocated()) {
        MPCR_PRINTER("Total memory allocated on GPU in bytes : ")
        MPCR_PRINTER(size)
        MPCR_PRINTER(std::endl)
    } else {
        MPCR_PRINTER("No allocations on GPU")
        MPCR_PRINTER(std::endl)
    }

}


/** ------------------------- Operators for R ---------------------------------- **/

DataType *
DataType::PerformPlusDispatcher(SEXP aObj) {

    if (TYPEOF(aObj) == REALSXP) {
        auto val = Rcpp::as <double>(aObj);
        return RPerformPlus(this, val, "");

    } else if (TYPEOF(aObj) == VECSXP || TYPEOF(aObj) == INTSXP) {
        auto values = Rcpp::as <std::vector <double>>(aObj);
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return RPerformPlus(this, &temp_mpr);

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
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return RPerformPow(this, &temp_mpr);

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
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return RPerformDiv(this, &temp_mpr);

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
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return RPerformMult(this, &temp_mpr);

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
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return RPerformMinus(this, &temp_mpr);

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
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return RGreaterThan(this, &temp_mpr);

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
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return RGreaterThanOrEqual(this, &temp_mpr);

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
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return RLessThan(this, &temp_mpr);

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
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return RLessThanOrEqual(this, &temp_mpr);

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
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return REqual(this, &temp_mpr);

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
        DataType temp_mpr(0, DOUBLE);
        temp_mpr.Allocate(values, CPU);
        return RNotEqual(this, &temp_mpr);

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


/** ------------------- DISPATCHERS & Template functions ------------------- **/


template <typename T>
void
DataType::SumDispatcher(double &aResult) {
    aResult = 0;
    auto pData = (T *) this->GetData(CPU);
    for (auto i = 0; i < this->mSize; i++) {
        aResult += pData[ i ];
    }
}


template <typename T>
void
DataType::SquareSumDispatcher(double &aResult) {
    aResult = 0;
    auto pData = (T *) this->GetData(CPU);
    for (auto i = 0; i < this->mSize; i++) {
        aResult += pow(pData[ i ], 2);
    }
}


template <typename T>
void
DataType::ProductDispatcher(double &aResult) {
    aResult = 1;
    auto pData = (T *) this->GetData(CPU);
    for (auto i = 0; i < this->mSize; i++) {
        aResult *= pData[ i ];
    }
}


template <typename T>
void
DataType::DeterminantDispatcher(double &aResult) {

    double det = 1.0;
    auto data = (T *) this->GetData(CPU);
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
            if (std::abs(pData[ j * size + i ]) >
                std::abs(pData[ max_row * size + i ])) {
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


template <typename T>
void DataType::FillTriangleDispatcher(const double &aValue,
                                      const bool &aUpperTriangle) {

    auto row = this->GetNRow();
    auto col = this->GetNCol();
    auto pData = (T *) this->GetData(CPU);

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

    this->SetData((char *) pData, CPU);

}


template <typename T>
void
DataType::TransposeDispatcher() {

    auto pData = (T *) this->GetData(CPU);
    auto pOutput = (T *) mpcr::memory::AllocateArray(this->GetSizeInBytes(),
                                                     CPU,
                                                     nullptr);
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

    this->SetData((char *) pOutput, CPU);
    this->SetDimensions(col, row);

}


template <typename T>
void DataType::ConvertToRMatrixDispatcher(Rcpp::NumericMatrix *&aOutput) {

    auto pData = (T *) this->GetData(CPU);
    aOutput = new Rcpp::NumericMatrix(this->mpDimensions->GetNRow(),
                                      this->mpDimensions->GetNCol(), pData);

}


template <typename T>
void DataType::CheckNA(std::vector <int> &aOutput, Dimensions *&apDimensions) {
    auto pData = (T *) this->GetData(CPU);
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


template <typename T>
void
DataType::ConvertToVector(std::vector <double> &aOutput) {
    auto pData = (T *) this->GetData(CPU);
    aOutput.clear();
    aOutput.resize(this->mSize);
    aOutput.assign(pData, pData + this->mSize);
}


template <typename T>
void
DataType::ConvertPrecisionDispatcher(const Precision &aPrecision) {
    this->mPrecision = aPrecision;

    if (this->mSize == 0) {
        return;
    }
    switch (aPrecision) {
        case HALF: {
#ifdef USE_CUDA
            mData.GetDataPointer(GPU);
            mData.FreeMemory(CPU);
            mData.ChangePrecision <T, float16>();

#else
            MPCR_PRINTER("Half Precision is not supported, Converting automatically to single")
            MPCR_PRINTER(std::endl)
            this->ConvertPrecisionDispatcher <T>(FLOAT);
#endif

            break;
        }
        case FLOAT: {
            mData.ChangePrecision <T, float>();
            break;
        }
        case DOUBLE: {
            mData.ChangePrecision <T, double>();
            break;
        }
        default: {
            MPCR_API_EXCEPTION("Invalid Precision : Not Supported", -1);
        }
    }

}


template <typename T>
void
DataType::CheckNA(const size_t &aIndex, bool &aFlag) {
    T *data = (T *) GetData(CPU);
    aFlag = std::isnan(data[ aIndex ]);
}


template <typename T>
void
DataType::SetValue(size_t aIndex, double &aVal) {

    T *data = (T *) this->GetData(CPU);
    data[ aIndex ] = (T) aVal;
    this->SetData((char *) data, CPU);
}


template <typename T>
void
DataType::GetValue(size_t aIndex, double &aOutput) {
    auto pdata = (T *) this->GetData(CPU);
    aOutput = (double) ( pdata[ aIndex ] );
}


template <typename T>
void
DataType::PrintVal() {
    std::stringstream ss;
    auto stream_size = 10000;
    T *temp = (T *) this->GetData(CPU);

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
                MPCR_PRINTER(ss.str())
                ss.clear();
            }
        }
        if (print_rows * print_col != this->mSize) {
            ss << "Note Only Matrix with size 100*13 is printed" <<
               std::endl;
        }

        MPCR_PRINTER(std::string(ss.str()))
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
                    MPCR_PRINTER(std::string(ss.str()))
                    ss.clear();
                }
            }
        }
        ss << std::endl;
        MPCR_PRINTER(std::string(ss.str()))
    }

}


template <typename T>
void
DataType::PrintRowsDispatcher(const size_t &aRowIdx,
                              std::stringstream &aRowAsString) {

    auto pData = (T *) this->GetData(CPU);
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
DataType::Init(std::vector <double> *aValues,
               const OperationPlacement &aOperationPlacement) {
    if (this->mSize == 0) {
        return;
    }

    auto context = mpcr::kernels::ContextManager::GetOperationContext();
    if (aOperationPlacement == GPU && context->GetOperationPlacement() == CPU) {
        context = mpcr::kernels::ContextManager::GetGPUContext();
    }


    this->mData.Allocate(this->GetSizeInBytes(), aOperationPlacement);

    auto pData = (T *) this->mData.GetDataPointer(aOperationPlacement);

    if (aValues == nullptr) {
        auto pdata_temp_char = this->mData.GetDataPointer(aOperationPlacement);
        mpcr::memory::Memset(pdata_temp_char, 0, this->GetSizeInBytes(),
                             aOperationPlacement, context);

        this->mData.SetDataPointer(pdata_temp_char, this->GetSizeInBytes(),
                                   aOperationPlacement);

    } else {
        if (aValues->size() < this->mSize) {
            auto idx = aValues->size();
            aValues->resize(mSize);
            for (auto i = idx; i < this->mSize; i++) {
                aValues->at(i) = 0;
            }
        }
        if (aOperationPlacement == CPU) {
            std::copy(aValues->begin(), aValues->end(), pData);
        } else {
#ifdef USE_CUDA

            auto size_in_bytes = aValues->size() * sizeof(double);
            auto temp_data = mpcr::memory::AllocateArray(size_in_bytes, GPU,
                                                         context);

            mpcr::memory::MemCpy(temp_data, (char *) aValues->data(),
                                 size_in_bytes, context,
                                 mpcr::memory::MemoryTransfer::HOST_TO_DEVICE);

            mpcr::memory::CopyDevice <double, T>((char *) temp_data,
                                                 (char *) pData,
                                                 aValues->size());

            this->mData.SetDataPointer((char *) pData, this->GetSizeInBytes(),
                                       aOperationPlacement);
            mpcr::memory::DestroyArray(temp_data, GPU, context);

#else
            MPCR_API_EXCEPTION("Cannot Perform GPU Allocation, No GPU Support", -1);
#endif
        }
    }


}


void
DataType::CheckHalfCompatibility(const OperationPlacement &aOperationPlacement) {
    if (mPrecision == HALF && aOperationPlacement == CPU) {
        MPCR_PRINTER("CPU doesn't support 16-bit, ")
        MPCR_PRINTER("the data will be converted to 32-bit")
        MPCR_PRINTER(std::endl)
        SIMPLE_DISPATCH_WITH_HALF(this->mPrecision, ConvertPrecisionDispatcher,
                                  FLOAT)
    }
}

/** ------------------------- INSTANTIATIONS ---------------------------------- **/

SIMPLE_INSTANTIATE(void, DataType::DeterminantDispatcher, double &aResult)

SIMPLE_INSTANTIATE(void, DataType::ProductDispatcher, double &aResult)

SIMPLE_INSTANTIATE(void, DataType::SumDispatcher, double &aResult)

SIMPLE_INSTANTIATE(void, DataType::SquareSumDispatcher, double &aResult)

SIMPLE_INSTANTIATE(void, DataType::FillTriangleDispatcher, const double &aValue,
                   const bool &aUpperTriangle)

SIMPLE_INSTANTIATE(void, DataType::CheckNA, std::vector <int> &aOutput,
                   Dimensions *&apDimensions)

SIMPLE_INSTANTIATE(void, DataType::CheckNA, const size_t &aIndex, bool &aFlag)

SIMPLE_INSTANTIATE(void, DataType::PrintVal)


SIMPLE_INSTANTIATE(void, DataType::GetValue, size_t aIndex, double &aOutput)

SIMPLE_INSTANTIATE(void, DataType::SetValue, size_t aIndex, double &aVal)

SIMPLE_INSTANTIATE(void, DataType::ConvertToVector,
                   std::vector <double> &aOutput)

SIMPLE_INSTANTIATE(void, DataType::ConvertToRMatrixDispatcher,
                   Rcpp::NumericMatrix *&aOutput)

SIMPLE_INSTANTIATE(void, DataType::TransposeDispatcher)

SIMPLE_INSTANTIATE(void, DataType::PrintRowsDispatcher, const size_t &aRowIdx,
                   std::stringstream &aRowAsString)

SIMPLE_INSTANTIATE_WITH_HALF(void, DataType::ConvertPrecisionDispatcher,
                             const Precision &aPrecision)

SIMPLE_INSTANTIATE_WITH_HALF(void, DataType::Init,
                             std::vector <double> *aValues,
                             const OperationPlacement &aOperationPlacement)

