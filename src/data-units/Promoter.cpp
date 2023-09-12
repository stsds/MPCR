/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <data-units/Promoter.hpp>


void
Promoter::Promote() {

    if (mCounter != mPrecisions.size()) {
        MPCR_API_EXCEPTION("Cannot Promote without inserting all elements", -1);
    }

    Precision highest_precision = mpcr::precision::FLOAT;

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
        if (mPrecisions[ i ] == mpcr::precision::HALF) {
            mDataHolders[ i ]->ConvertPrecision(mPrecisions[ i ]);
        }
    }
}


DataType *
Promoter::GetPromotedTile(DataType *&apTile,
                          const Precision &aPrecisionRequired) {

    if (apTile->GetPrecision() == aPrecisionRequired) {
        return apTile;
    }

    if (mTileMap.find(apTile) != mTileMap.end()) {
        auto temp_tiles = mTileMap[ apTile ];
        for (auto &tile: temp_tiles) {
            if (tile->GetPrecision() == aPrecisionRequired) {
                return tile;
            }
        }

    }

    auto pTemp_tile = new DataType(*apTile, aPrecisionRequired);
    mTileMap[ apTile ].push_back(pTemp_tile);
    return pTemp_tile;

}


void
Promoter::ResetPromoter(const size_t &aCount) {
    mPrecisions.clear();
    mDataHolders.clear();

    mPrecisions.resize(aCount);
    mDataHolders.resize(aCount);
    mCounter = 0;

    if (!mTileMap.empty()) {
        for (auto &vec_tiles: mTileMap) {
            for (auto &tile: vec_tiles.second) {
                delete tile;
            }
            vec_tiles.second.clear();
        }
    }
    mTileMap.clear();
}

