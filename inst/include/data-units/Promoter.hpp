
#ifndef MPR_PROMOTER_HPP
#define MPR_PROMOTER_HPP

#include <data-units/DataType.hpp>


using namespace mpr::precision;

class Promoter {

public:

    /**
     * @brief
     * Constructor
     *
     * @param[in] aCount
     * Number of MPR objects that will be used for Promotion Process
     *
     */
    Promoter(int aCount) {
        mPrecisions.resize(aCount);
        mDataHolders.resize(aCount);
        mCounter = 0;
    };

    /**
     * @brief
     * Default De-Constructor
     *
     */
    ~Promoter() = default;


    /**
     * @brief
     * Insert MPR Objects to use for Promotion
     *
     * @param[in] aInput
     * MPR Object to insert
     *
     */
    inline
    void
    Insert(DataType &aInput) {
        mPrecisions[ mCounter ] = aInput.GetPrecision();
        mDataHolders[ mCounter ] = &aInput;
        mCounter++;
    }


    /**
     * @brief
     * Promote all the inserted MPR Objects according to the Highest Object
     * Precision
     *
     */
    void
    Promote();

    /**
     * @brief
     * De-Promotes the Half Precision (ONLY) objects that were promoted.
     * can be extended to de-promote all MPR objects to their original Precision.
     *
     * Note:
     * No MPR Object pointer should be changed in any process in between Promotion
     * and De-Promotion.
     *
     */
    void
    DePromote();

    inline
    void
    ResetPromoter(const size_t &aCount){
        mPrecisions.clear();
        mDataHolders.clear();

        mPrecisions.resize(aCount);
        mDataHolders.resize(aCount);
        mCounter = 0;
    }


private:
    /** vector of precisions of MPR objects before any promotion **/
    std::vector <Precision> mPrecisions;
    /** vector of pointers to the original MPR objects **/
    std::vector <DataType *> mDataHolders;
    /** Number of object currently inserted in the promoter **/
    int mCounter;
};


#endif //MPR_PROMOTER_HPP
