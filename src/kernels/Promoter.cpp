/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <kernels/Promoter.hpp>

using namespace mpcr::kernels;

void
Promoter::Promote(const Precision &aOperationLowestPrecision) {

    if (mCounter != mPrecisions.size()) {
        MPCR_API_EXCEPTION("Cannot Promote without inserting all elements", -1);
    }

    Precision highest_precision = aOperationLowestPrecision;

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
        if (mPrecisions[ i ] == mpcr::definitions::HALF) {
            mDataHolders[ i ]->ConvertPrecision(mPrecisions[ i ]);
        }
    }
}

void
Promoter::ResetPromoter(const size_t &aCount) {
    mPrecisions.clear();
    mDataHolders.clear();

    mPrecisions.resize(aCount);
    mDataHolders.resize(aCount);
    mCounter = 0;
}

