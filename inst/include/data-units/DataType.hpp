

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
    Dimensions(Dimensions &aDimensions) {
        this->mRow = aDimensions.mRow;
        this->mRow = aDimensions.mRow;
    }


    /**
     * @brief
     * Dimensions Copy Constructor
     *
     * @param[in] aDimensions
     * Dimensions struct object to copy its dimensions
     */
    Dimensions &
    operator=(const Dimensions &aDimensions) {
        this->mRow = aDimensions.mRow;
        this->mCol = aDimensions.mCol;
        return *this;
    }


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


/** DataType Class creates an array of float that you can access throw
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
    DataType(size_t aSize, std::string aPrecision);

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
    operator=(DataType &aDataType);

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
        if (this->mpData != nullptr) {
            delete this->mpData;
            this->mpData = nullptr;
        }
        if (this->mpDimensions != nullptr) {
            delete this->mpDimensions;
            this->mpDimensions = nullptr;
        }
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
    GetNRow();

    /**
     * @brief
     * Get Number of Columns
     *
     * @returns
     * Number of Columns in a Matrix
     *
     */
    size_t
    GetNCol();

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
    GetDimensions();


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
    template<typename T>
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
    template<typename T>
    void
    SetValue(int aIndex, double &aVal);

    /**
     * @brief
     * Prints all data in the Vector according to its type
     */
    template<typename T>
    void
    PrintVal();


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
        return ((aRow * aCol) == this->mSize);
    }


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
    template<typename T>
    void
    GetCopyOfData(char *&aSrc, char *&aDest);

    /**
     * @brief
     * Initialize Data Buffer according to its type
     */
    template<typename T>
    void
    Init();

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
};

#endif //MPR_DATATYPE_HPP
