

#include <data-units/Promoter.hpp>


void
Promoter::Promote() {

    if (mCounter != mPrecisions.size()) {
        MPR_API_EXCEPTION("Cannot Promote without inserting all elements", -1);
    }

    Precision highest_precision = mpr::precision::FLOAT;

    for (auto &x: mPrecisions) {
        if (x > highest_precision) {
            highest_precision = x;
        }
    }

    for (auto &x: mDataHolders) {
        x->ConvertPrecision(highest_precision);
    }

}


void
Promoter::DePromote() {

    for (auto i = 0; i < mCounter; i++) {
        if (mPrecisions[ i ] == mpr::precision::HALF) {
            mDataHolders[ i ]->ConvertPrecision(mPrecisions[ i ]);
        }
    }
}