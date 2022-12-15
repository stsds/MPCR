

#ifndef MPR_DATATYPE_HPP
#define MPR_DATATYPE_HPP


#include <vector>
#include <string>
#include <Rcpp.h>


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
     * @param[in] aVal
     * value to initialize array with
     *
     */
    DataType(size_t aSize, float aVal);


    /**
     * @brief
     * DataType de-constructor
     *
     */
    ~DataType() {
        if (mpData != nullptr) {
            delete mpData;
        }
    }


    /**
     * @brief
     * Initialize Float array with a value
     *
     * @param[in] aSize
     * Size of Vector
     * @param[in] aVal
     * value to initialize array with
     *
     */
    void
    DataTypeInit(size_t aSize, float aVal);


    /**
     * @brief
     * Print all values in vector
     */
    void
    PrintVal();

    /**
     * @brief
     * Get Float pointer array containing mpData
     *
     * @return
     * float pointer array
     *
     */
    float *
    GetData();

    /**
     * @brief
     * Get david of array
     *
     * @return
     * array david
     *
     */
    size_t
    GetSize();


private:
    /** Float pointer array that contains Data **/
    float *mpData;
    /** Size of Float Pointer Array **/
    size_t mVecSize;
};

#endif //MPR_DATATYPE_HPP
