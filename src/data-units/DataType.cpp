
#include <data-units/DataType.hpp>
#include <utilities/MPRDispatcher.hpp>
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
    T *temp = (T *) this->mpData;
    Rcpp::Rcout << mSize << std::endl;
    Rcpp::Rcout << "---------------------" << std::endl;
    for (auto i = 0; i < mSize; i++) {
        Rcpp::Rcout << temp[ i ] << std::endl;
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
DataType::GetValue(int aIndex, double *&aOutput) {
    double *temp = new double;
    *temp = (double) (((T *) this->mpData )[ aIndex ] );
    delete aOutput;
    aOutput = temp;
}


double
DataType::GetVal(int aIndex) {
    if (aIndex >= this->mSize) {
        MPR_API_EXCEPTION("Segmentation Fault Index Out Of Bound", -1);
    }
    double *temp = nullptr;
    SIMPLE_DISPATCH(mPrecision, GetValue, aIndex, temp)
    return *temp;
}


template <typename T>
void
DataType::SetValue(int aIndex, double &aVal) {

    T *data = (T *) this->mpData;
    data[ aIndex ] = (T) aVal;
}


void
DataType::SetVal(int aIndex, double aVal) {
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
DataType::GetCopyOfData(const char *aSrc, char *&aDest) {
    T *data = (T *) aSrc;
    auto size = this->mSize;
    T *output = new T[size];

    memcpy((char *) output, (char *) data, size * sizeof(T));
    aDest = (char *) output;
}


DataType &
DataType::operator =(const DataType &aDataType) {
    this->mSize = aDataType.mSize;
    this->mPrecision = aDataType.mPrecision;
    this->mMatrix = aDataType.mMatrix;
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
    SetVal(idx,aVal);
}


void DataType::SetMagicNumber() {
    this->mMagicNumber=911;
}

SIMPLE_INSTANTIATE(void, DataType::CheckNA, const size_t &aIndex, bool &aFlag)

SIMPLE_INSTANTIATE(void, DataType::Init)

SIMPLE_INSTANTIATE(void, DataType::PrintVal)

SIMPLE_INSTANTIATE(void, DataType::GetCopyOfData, const char *aSrc,
                   char *&aDest)

SIMPLE_INSTANTIATE(void, DataType::GetValue, int aIndex, double *&aOutput)

SIMPLE_INSTANTIATE(void, DataType::SetValue, int aIndex, double &aVal)

SIMPLE_INSTANTIATE(void, DataType::GetDataSize, size_t &aDataSize)