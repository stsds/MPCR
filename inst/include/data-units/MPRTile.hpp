

#ifndef MPR_MPRTILE_HPP
#define MPR_MPRTILE_HPP

#include <data-units/DataType.hpp>


/** Pair describing a Matrix Index using row,col **/
typedef std::pair <size_t, size_t> MatrixIndex;

class MPRTile {

public:

    /**
     * @brief
     * MPR-Tile Constructor
     *
     * @param[in] aRow
     * Number of Global rows
     * @param[in] aCol
     * Number of Global cols
     * @param[in] aTileRow
     * Number of rows in each tile
     * @param[in] aTileCol
     * Number of cols in each tile
     * @param[in] aValues
     * vector containing all the data needed to construct the Matrix from.
     * should be the size of row*col
     * @param[in] aPrecisions
     * vector indicating the precision of each tile
     * should be the same size as the required tiles
     *
     */
    MPRTile(size_t aRow, size_t aCol, size_t aTileRow, size_t aTileCol,
            const std::vector <double> &aValues,
            const std::vector <std::string> &aPrecisions);

    /**
     * @brief
     * MPR-Tile Constructor , Creates an Empty MPRTile Object with the right
     * Dimensions (All Tiles pointers are Null)
     *
     * @param[in] aRow
     * Number of Global rows
     * @param[in] aCol
     * Number of Global cols
     * @param[in] aTileRow
     * Number of rows in each tile
     * @param[in] aTileCol
     * Number of cols in each tile
     *
     */
    MPRTile(size_t aRow, size_t aCol, size_t aTileRow, size_t aTileCol);

    /**
     * @brief
     * MPR-Tile Equal Operator
     *
     * @param[in] aMPRTile
     * MPRTile Object.
     *
     * @returns
     * Deep Copy of MPRTile Object provided.
     *
     */
    MPRTile &
    operator =(const MPRTile &aMPRTile);

    /**
     * @brief
     * MPR-Tile Copy Constructor
     *
     * @param[in] aMPRTile
     * MPRTile Object.
     *
     */
    MPRTile(const MPRTile &aMPRTile);


    /**
     * @brief
     * MPR-Tile De-Constructor
     *
     */
    ~MPRTile() {
        for (auto &x: mTiles) {
            delete x;
        }
        mTiles.clear();
        delete this->mpTilesDimensions;
        delete this->mpDimensions;
        delete this->mpTileInnerDimensions;
    }


    /**
     * @brief
     * Sets the value at [aRowIdx,aColIdx]
     *
     * @param[in] aRowIdx
     * Idx row in Global Matrix
     * @param[in] aColIdx
     * Idx col in Global Matrix
     * @param[in] aValue
     * value to set matrix element with
     *
     */
    void
    SetVal(const size_t &aRowIdx, const size_t &aColIdx, double aValue);

    /**
     * @brief
     * Get the value at [aRowIdx,aColIdx]
     *
     * @param[in] aRowIdx
     * Idx row in Global Matrix
     * @param[in] aColIdx
     * Idx col in Global Matrix
     * @returns
     * value at [aRowIdx,aColIdx] matrix element with
     *
     */
    double
    GetVal(const size_t &aRowIdx, const size_t &aColIdx);


    /**
     * @brief
     * Get Number of Rows in Global Matrix
     *
     * @returns
     * Number of Rows in Global Matrix
     *
     */
    inline
    size_t
    GetNRow() const {
        return this->mpDimensions->GetNRow();
    }


    /**
     * @brief
     * Get Number of Cols in Global Matrix
     *
     * @returns
     * Number of Cols in Global Matrix
     *
     */
    inline
    size_t
    GetNCol() const {
        return this->mpDimensions->GetNCol();
    }


    /**
     * @brief
     * Get Number of Rows in Tiles
     *
     * @returns
     * Number of Rows in Tiles
     *
     */
    inline
    size_t
    GetTileNRow() const {
        return this->mpTileInnerDimensions->GetNRow();
    }


    /**
     * @brief
     * Get Number of Cols in Tiles
     *
     * @returns
     * Number of Cols in Tiles
     *
     */
    inline
    size_t
    GetTileNCol() const {
        return this->mpTileInnerDimensions->GetNCol();
    }


    /**
     * @brief
     * Get Number of Tiles per Col
     *
     * @returns
     * Number of Tiles per Col
     *
     */
    inline
    size_t
    GetTilePerCol() const {
        return this->mpTilesDimensions->GetNCol();
    }


    /**
     * @brief
     * Get Number of Tiles per Row
     *
     * @returns
     * Number of Tiles per Row
     *
     */
    inline
    size_t
    GetTilePerRow() const {
        return this->mpTilesDimensions->GetNRow();
    }


    /**
     * @brief
     * Change Precision of a certain tile
     *
     * @param[in] aTileRowIdx
     * Row idx of tile
     * @param[in] aTileColIdx
     * Col idx of tile
     * @param[in] aPrecision
     * Precision Required
     *
     */
    void
    ChangePrecision(const size_t &aTileRowIdx, const size_t &aTileColIdx,
                    const mpr::precision::Precision &aPrecision);


    /**
     * @brief
     * Change Precision of a certain tile
     *
     * @param[in] aTileRowIdx
     * Row idx of tile
     * @param[in] aTileColIdx
     * Col idx of tile
     * @param[in] aPrecision
     * Precision Required
     *
     */
    void
    ChangePrecision(const size_t &aTileRowIdx, const size_t &aTileColIdx,
                    const std::string &aPrecision);


    /**
     * @brief
     * Print the values of a certain tile
     *
     * @param[in] aTileRowIdx
     * Row idx of tile
     * @param[in] aTileColIdx
     * Col idx of tile
     *
     */
    void
    PrintTile(const size_t &aTileRowIdx, const size_t &aTileColIdx);


    /**
     * @brief
     * Return Matrix described as a vector of DataType Matrices in a
     * Column Major Form
     *
     * @return
     * Vector of DataTypes Pointers
     *
     */
    inline
    std::vector <DataType *> &
    GetTiles() {
        return this->mTiles;
    };


    /**
     * @brief
     * Get Size of Tile
     *
     * @returns
     * Size of Tile
     *
     */
    inline
    size_t
    GetTileSize() {
        return this->mTileSize;
    }


    /**
     * @brief
     * Get Size of Matrix
     *
     * @returns
     * Size of Matrix
     *
     */
    inline
    size_t
    GetMatrixSize() {
        return this->mSize;
    }


    /**
     * @brief
     * Check if a Casted Memory Address is a MPRTile Object.
     *
     * @returns
     * true if the casted pointer is a MPRTile Object, False Otherwise
     */
    inline
    const bool
    IsMPRTile() const {
        return ( this->mMagicNumber == 505 );
    }


    /**
     * @brief
     * Prints the Metadata of the MPRTile Object
     *
     */
    void
    GetType();

    /**
     * @brief
     * Prints the Metadata and the values of the MPRTile Object
     *
     */
    void
    Print();

    /**
     * @brief
     * Insert a Tile inside the MPR Tiles using Tile Row idx and Tile Col idx.
     * this function doesn't check whether the internal dimensions of the tile
     * is the same as the Dimensions used to created the Tiled Matrix
     *
     * @param[in] apTile
     * MPR Tile to insert
     * @param[in] aTileRowIdx
     * Row Idx of the Tile
     * @param[in] aTileColIdx
     * Col Idx of the Tile
     */
    void
    InsertTile(DataType *apTile, const size_t &aTileRowIdx,
               const size_t &aTileColIdx);


    /**
     * @brief
     * Get a Tile from the MPR Tiles using Tile Row idx and Tile Col idx.
     *
     * @param[in] aTileRowIdx
     * Row Idx of the Tile
     * @param[in] aTileColIdx
     * Col Idx of the Tile
     *
     * @returns
     * MPR object describing the Tile at IDx (aTileRowIdx,aTileColIdx)
     */
    DataType *&
    GetTile(const size_t &aTileRowIdx, const size_t &aTileColIdx);

    /**
     * @brief
     * Copies the MPRTile Dimensions and sizes metadata from another MPRTile obj
     *
     * @param[in] aMPRTile
     * MPRTile Object.
     */
    void
    SetDimensions(MPRTile &aMPRTile);


    /**
     * @brief
     * Convert a 2D Index into 1D Index Column Major, Given the leading Dimension
     * used for the conversion
     *
     * @param[in] aIdx
     * pair of idx [Row,Col]
     * @param[in] aLeadingDim
     * Leading Dimension to use for conversion to a 1D idx
     *
     * @returns
     * Idx in a 1D Form
     *
     */
    inline
    size_t
    GetIndexColumnMajor(const MatrixIndex &aIdx,
                        const size_t &aLeadingDim) {
        return ( aIdx.second * aLeadingDim ) + aIdx.first;
    }


    /**
     * @brief
     * Fills the Null Tiles with Zeros according to the MPRTile Metadata and
     * Dimensions.
     *
     */
    void
    FillWithZeros();

    /**
     * @brief
     * Fills Upper or Lower Triangle with a given value and precision
     * New Tiles will be created ,replacing the old tiles
     * Note:
     * the Input must be a square Matrix
     *
     * @param[in] aValue
     * value to use for filling the triangle
     * @param[in] aUpperTriangle
     * bool to indicate whether to fill the upper or the lower triangle
     * if true, the upper triangle will be accessed ;otherwise, the lower
     * @param[in] aPrecision
     * Default precision to use for filling the triangle
     *
     */
    void
    FillSquareTriangle(const double &aValue, const bool &aUpperTriangle = true,
                       const mpr::precision::Precision &aPrecision = mpr::precision::FLOAT);

private:

    /**
     * @brief
     * Determines which Tile contains the Given Idxs
     *
     * @param[in] aMatrixIndex
     * pair of idx [Row,Col] Global Idx
     *
     * @returns
     * Pair of idx [Row,Col] of Tile
     *
     */
    MatrixIndex
    GetTileIndex(const MatrixIndex &aMatrixIndex);

    /**
     * @brief
     * Determines the local index inside a certain tile given the global idx
     *
     * @param[in] aIdxGlobal
     * pair of idx [Row,Col] Global Idx
     * @param[in] aTileIdx
     * pair of idx [Row,Col] Tile Idx
     *
     * @returns
     * Pair of idx [Row,Col] of element inside the tile
     *
     */
    MatrixIndex
    GetLocalIndex(const MatrixIndex &aIdxGlobal, const MatrixIndex &aTileIdx);

    /**
     * @brief
     * Determines the Global index Given a certain tile and Idx of element inside
     * the tile
     *
     * @param[in] aIdxLocal
     * pair of idx [Row,Col] Local Idx
     * @param[in] aTileIdx
     * pair of idx [Row,Col] Tile Idx
     *
     * @returns
     * Pair of idx [Row,Col] of element inside the Global Matrix
     *
     */
    MatrixIndex
    GetGlobalIndex(const MatrixIndex &aIdxLocal, const MatrixIndex &aTileIdx);


    /**
     * @brief
     * Set Magic Number To Check For MPRTile Object.
     *
     */
    inline
    void
    SetMagicNumber() {
        this->mMagicNumber = 505;
    }


    /**
     * @brief
     * Allocate Tile data with the corresponding data from values given
     *
     * @param[in] aTile
     * DataType object representing a Tile
     * @param[in] aTileRowIdx
     * Row idx of the Tile
     * @param[in] aTileColIdx
     * Col idx of the Tile
     * @param[in] aValues
     * Vector holding all the values of the global matrix that should be assigned
     * to the tiles.
     *
     */
    template <typename T>
    void
    AssignValuesToTile(DataType &aTile, const size_t &aTileRowIdx,
                       const size_t &aTileColIdx,
                       const std::vector <double> &aValues);

    /**
     * @brief
     * Checks if a Given Row and Col Idx are in range of a certain Dimensions
     *
     * @param[in] aRowIdx
     * Row Idx
     * @param[in] aColIdx
     * Col Idx
     * @param[in] aDimensions
     * Dimension object to compare idx with
     *
     * @returns
     * true if idx are in range , false otherwise
     */
    bool
    CheckIndex(const size_t &aRowIdx, const size_t &aColIdx,
               const Dimensions &aDimensions);

    /**
     * @brief
     * Checks if a MPRTile object can be created from the given dimensions, if
     * yes , The function will set all the metadata required for the MPRTile
     * object.
     *
     * @param[in] aRow
     * Number of Rows of Global Matrix
     * @param[in] aCol
     * Number of Cols of Global Matrix
     * @param[in] aTileRow
     * Number of Rows of Each Tile Matrix
     * @param[in] aTileCol
     * Number of Cols of Each Tile Matrix
     *
     */
    void
    AssignDimensions(const size_t &aRow, const size_t &aCol,
                     const size_t &aTileRow, const size_t &aTileCol);

private:

    /**
     * Vector Containing datatypes object , each represent a separate tile
     * in column major for
     **/
    std::vector <DataType *> mTiles;
    /** Dimensions object that describe the Global Matrix in General **/
    Dimensions *mpDimensions;
    /** Dimensions object that describe the Global Matrix in Tiles form **/
    Dimensions *mpTilesDimensions;
    /** Dimensions object that describe the Tiles Matrix Dimensions **/
    Dimensions *mpTileInnerDimensions;
    /** Total Size of the Global Matrix **/
    size_t mSize;
    /** Size of Tile **/
    size_t mTileSize;
    /** Magic Number to check if object is MPR Tile **/
    int mMagicNumber;

};

#endif //MPR_MPRTILE_HPP
