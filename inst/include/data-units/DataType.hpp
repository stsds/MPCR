

#ifndef MPR_DATATYPE_HPP
#define MPR_DATATYPE_HPP


#include <vector>
#include <data-units/Precision.hpp>


/** Dimensions struct holding Dimensions for Representing a Vector as a Matrix **/
typedef struct Dimensions {

    /** size_t representing number of Rows in a Matrix**/
    size_t mCol;
    /** size_t representing number of Columns in a Matrix**/
    size_t mRow;

    /**
     * @brief
     * Default Constructor
     */
    Dimensions() = default;


    /**
     * @brief
     * Dimensions Constructor
     *
     * @param[in] aRow
     * Number of Rows in a Matrix
     * @param[in] aCol
     * Number of Columns in a Matrix
     *
     */
    explicit
    Dimensions(size_t aRow, size_t aCol) {
        this->mRow = aRow;
        this->mCol = aCol;
    }


    /**
     * @brief
     * Dimensions Constructor
     *
     * @param[in] aDimensions
     * Dimensions struct object to copy its dimensions
     */
    Dimensions(const Dimensions &aDimensions) {
        this->mRow = aDimensions.mRow;
        this->mCol = aDimensions.mCol;
    }


    /**
     * @brief
     * Dimensions Copy Constructor
     *
     * @param[in] aDimensions
     * Dimensions struct object to copy its dimensions
     */
    Dimensions &
    operator =(const Dimensions &aDimensions) = default;


    /**
     * @brief
     * Get Number of Rows
     *
     * @returns
     * Number of Rows in a Matrix
     */
    size_t
    GetNRow() const {
        return this->mRow;
    }


    /**
     * @brief
     * Get Number of Columns
     *
     * @returns
     * Number of Columns in a Matrix
     */
    size_t
    GetNCol() const {
        return this->mCol;
    }


    /**
     * @brief
     * Set Number of Columns
     *
     * @param[in] aCol
     * Number of Columns in a Matrix
     */
    void
    SetNCol(size_t aCol) {
        this->mCol = aCol;
    }


    /**
     * @brief
     * Set Number of Rows
     *
     * @param[in] aRow
     * Number of Rows in a Matrix
     */
    void
    SetNRow(size_t aRow) {
        this->mRow = aRow;
    }


    /**
     * @brief
     * Default De-Constructor
     */
    ~Dimensions() = default;

} Dimensions;


/** DataType Class creates an array of (16/32/64)-Bit Precision that you can access throw
 * R as C++ object
 **/
class DataType {

public:

    /**
     * @brief
     * DataType Constructor
     *
     * @param[in] aSize
     * Size of Vector
     * @param[in] aPrecision
     * Precision to Describe the Values (as a Precision ENUM object)
     */
    DataType(size_t aSize, mpr::precision::Precision aPrecision);

    /**
     * @brief
     * DataType Copy Constructor
     *
     * @param[in] aDataType
     * DataType object to copy its content
     */
    DataType(DataType &aDataType);

    /**
     * @brief
     * DataType Constructor
     *
     * @param[in] aSize
     * Size of Vector
     * @param[in] aPrecision
     * Precision to Describe the Values (as a String)
     */
    DataType(size_t aSize, const std::string &aPrecision);

    /**
     * @brief
     * DataType Constructor
     *
     * @param[in] aRow
     * Number Of Rows in Matrix
     * @param[in] aCol
     * Number Of Columns in Matrix
     * @param[in] aPrecision
     * Precision to Describe the Values (as a Precision ENUM object)
     */
    DataType(size_t aRow, size_t aCol, mpr::precision::Precision aPrecision);

    /**
     * @brief
     * DataType Constructor
     *
     * @param[in] aSize
     * Size of Vector
     * @param[in] aPrecision
     * Precision to Describe the Values (as an int)
     */
    DataType(size_t aSize, int aPrecision);

    /**
     * @brief
     * DataType Constructor ,Creates Datatype Object with nothing more than a
     * precision initialized
     *
     * @param[in] aPrecision
     * Precision to Describe the Values (as a Precision ENUM object)
     */
    explicit
    DataType(mpr::precision::Precision aPrecision);

    /**
     * @brief
     * DataType Copy Operator
     *
     * @param[in] aDataType
     * DataType object to copy its content
     */
    DataType &
    operator =(const DataType &aDataType);

    /**
     * @brief
     * DataType De-Constructor
     */
    ~DataType();


    /**
     * @brief
     * DataType ClearUp Function to Clear all Pointers and data inside except
     * Precision
     */
    inline void
    ClearUp() {
        this->mSize = 0;
        this->mMatrix = false;
        delete[] this->mpData;
        delete this->mpDimensions;
        this->mpData = nullptr;
        this->mpDimensions = nullptr;
    }


    /**
     * @brief
     * Changes a Vector to a Matrix
     *
     * @param[in] aRow
     * Number Of Rows in Matrix
     * @param[in] aCol
     * Number Of Columns in Matrix
     */
    void
    ToMatrix(size_t aRow, size_t aCol);

    /**
     * @brief
     * Changes a Matrix to a Vector
     */
    void
    ToVector();

    /**
     * @brief
     * Prints all data in the Vector
     */
    void
    Print();

    /**
     * @brief
     * Get Values in the Vector according to Index (0-based Indexing)
     *
     * @param[in] aIndex
     * Index in vector
     *
     * @returns
     * Value with idx=aIndex in vector
     */
    double
    GetVal(int aIndex);

    /**
     * @brief
     * Get Values in the Matrix according to Index (0-based Indexing)
     *
     * @param[in] aRow
     * Row Index
     * @param[in] aCol
     * Col Index
     *
     * @returns
     * Value at idx [row][col]
     */
    double
    GetValMatrix(const size_t &aRow, const size_t &aCol);

    /**
     * @brief
     * Set Values in the Vector according to Index (0-based Indexing)
     *
     * @param[in] aIndex
     * Index in Vector
     * @param[in] aVal
     * Value used to set the vector[idx] with
     */
    void
    SetVal(int aIndex, double aVal);

    /**
     * @brief
     * Set Values in the Vector according to Index (0-based Indexing)
     *
     * @param[in] aRow
     * Row Index
     * @param[in] aCol
     * Col Index
     * @param[in] aVal
     * Value used to set the vector[idx] with
     */
    void
    SetValMatrix(size_t aRow, size_t aCol, double aVal);

    /**
     * @brief
     * Get Precision of the Object
     *
     * @returns
     * Precision Object
     */
    mpr::precision::Precision &
    GetPrecision();

    /**
     * @brief
     * Get Data of Vector
     *
     * @returns
     * Char pointer pointing to vector data (Must be casted according to precision)
     */
    char *
    GetData();

    /**
     * @brief
     * Get Size of Vector or Matrix
     *
     * @returns
     * Size of Vector or total size of Matrix
     */
    size_t
    GetSize() const;

    /**
     * @brief
     * Set Size of Vector or Matrix
     *
     * @param[in] aSize
     * Size of Matrix or Vector
     */
    void
    SetSize(size_t aSize);

    /**
     * @brief
     * Set Precision of Vector or Matrix (Currently it clears up all data)
     *
     * @param[in] aPrecision
     * Precision of Vector or Matrix
     */
    void
    SetPrecision(mpr::precision::Precision aPrecision);

    /**
     * @brief
     * bool indicating whether it's a Matrix or Vector
     *
     * @returns
     * True if it's a Matrix ,False otherwise
     */
    bool
    IsMatrix() const;

    /**
     * @brief
     * Get Index of Matrix in Row-Wise Representation
     *
     * @param[in] aRow
     * Index Of Row in Matrix
     * @param[in] aCol
     * Index Of Column in Matrix
     *
     * @returns
     * Index in 1D representation
     */
    size_t
    GetMatrixIndex(size_t aRow, size_t aCol);

    /**
     * @brief
     * Set Data Buffer. (Clears the Current Buffer)
     *
     * @param[in] aData
     * Buffer to set Object Buffer With.
     */
    void
    SetData(char *aData);

    /**
     * @brief
     * Get Number of Rows
     *
     * @returns
     * Number of Rows in a Matrix
     */
    size_t
    GetNRow() const;

    /**
     * @brief
     * Get Number of Columns
     *
     * @returns
     * Number of Columns in a Matrix
     *
     */
    size_t
    GetNCol() const;

    /**
     * @brief
     * Set Matrix Dimensions (throw Error in case the Dimensions can't cover all
     * the data available)
     *
     * @param[in] aRow
     * Number Of Rows in Matrix
     * @param[in] aCol
     * Number Of Columns in Matrix
     */
    void
    SetDimensions(size_t aRow, size_t aCol);

    /**
     * @brief
     * Get Matrix Dimensions
     *
     * @returns
     * Dimensions struct containing the Dimensions of the Matrix
     */
    Dimensions *
    GetDimensions() const;


    /**
     * @brief
     * Return whether this Dimensions can be used to convert Vector to Matrix
     *
     * @returns
     * true if the given Row and Col covers all the data in the buffer,
     * false otherwise
     *
     */
    inline
    bool
    CanBeMatrix(size_t aRow, size_t aCol) const {
        return (( aRow * aCol ) == this->mSize );
    }


    /**
     * @brief
     * Check Whether Element at index is NAN or Not
     *
     * @param[in] aIndex
     * Index of Element to check
     *
     * @returns
     * true if NAN,-NAN else Otherwise
     *
     */
    bool
    IsNA(const size_t &aIndex);

    /**
     * @brief
     * Get total size of Memory used by MPR Object
     *
     * @returns
     * Total size of Memory used by MPR Object
     *
     */
    size_t
    GetObjectSize();


    /**
     * @brief
     * Get Element at Idx from MPR Vector as MPR Object
     *
     * @param[in] aIndex
     * Index to Get Value from
     *
     * @returns
     * MPR Object holding element at idx
     *
     */
    inline
    DataType *
    GetElementVector(const size_t &aIndex) {
        auto element = GetVal(aIndex);
        auto output = new DataType(1, this->mPrecision);
        output->SetVal(0, element);
        return output;
    }


    /**
     * @brief
     * Get Element with Idx [row][col] from MPR Matrix as MPR Object
     *
     * @param[in] aRow
     * Row Idx
     * @param[in] aCol
     * Col Idx
     *
     * @returns
     * MPR Object holding element at idx
     *
     */
    inline
    DataType *
    GetElementMatrix(const size_t &aRow, const size_t &aCol) {
        auto index = GetMatrixIndex(aRow, aCol);
        auto element = GetVal(index);
        auto output = new DataType(1, this->mPrecision);
        output->SetVal(0, element);
        return output;
    }


    /**
     * @brief
     * Check if a Casted Memory Address is a DataType.
     *
     * @returns
     * true if the casted pointer is a DataType Object, False Otherwise
     */
    inline
    const bool
    IsDataType() {
        return ( this->mMagicNumber == 911 ) ? true : false;
    }


private:

    /**
     * @brief
     * Get Values in the Vector according to Index (0-based Indexing) and
     * Precision type
     *
     * @param[in] aIndex
     * Index in Vector
     * @param[out] aOutput
     * Value at idx
     *
     */
    template <typename T>
    void
    GetValue(int aIndex, double *&aOutput);

    /**
     * @brief
     * Set Values in the Vector according to Index (0-based Indexing) and
     * Precision type
     *
     * @param[in] aIndex
     * Index in Vector
     * @param[in] aVal
     * Value to set index with
     */
    template <typename T>
    void
    SetValue(int aIndex, double &aVal);

    /**
     * @brief
     * Prints all data in the Vector according to its type
     */
    template <typename T>
    void
    PrintVal();

    /**
     * @brief
     * Copies Data From Src buffer to Dest Buffer
     *
     * @params[in] aSrc
     * Buffer to copy from
     * @params[out] aDest
     * Buffer to copy to
     *
     */
    template <typename T>
    void
    GetCopyOfData(const char *apSrc, char *&apDest);

    /**
     * @brief
     * Initialize Data Buffer according to its type
     */
    template <typename T>
    void
    Init();

    /**
     * @brief
     * Check Whether Element at Index is NAN or Not
     *
     * @param[in] aIndex
     * Index of Element to check
     * @param[out] aFlag
     * True if NAN,-NAN else otherwise
     *
     */
    template <typename T>
    void
    CheckNA(const size_t &aIndex, bool &aFlag);

    /**
     * @brief
     * Get total size of Memory used by Data in MPR Object
     *
     * @param[out] aDataSize
     * Total size of Memory used by Data in MPR Object
     *
     */
    template <typename T>
    void
    GetDataSize(size_t &aDataSize);

    /**
     * @brief
     * Set Magic Number To Check For DataType Object.
     *
     */
    void
    SetMagicNumber();

    /** Buffer Holding the Data **/
    char *mpData;
    /** Dimensions object that describe the Vector as a Matrix **/
    Dimensions *mpDimensions;
    /** Total size of Vector or Matrix (Data Buffer) **/
    size_t mSize;
    /** Precision used to describe the data buffer **/
    mpr::precision::Precision mPrecision;
    /** Bool indicating whether it's a Matrix(True) or Vector(False) **/
    bool mMatrix;
    /** Magic Number to check if object is DataType **/
    int mMagicNumber;
};

#endif //MPR_DATATYPE_HPP