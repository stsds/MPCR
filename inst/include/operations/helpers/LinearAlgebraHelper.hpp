/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_LINEARALGEBRAHELPER_HPP
#define MPCR_LINEARALGEBRAHELPER_HPP

#include <data-units/DataType.hpp>
#include <lapack.hh>


template <typename T>
T
NormMACS(DataType &aInput) {
    T norm = 0.0f;
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();
    auto pData = (T *) aInput.GetData();

    for (auto j = 0; j < col; j++) {
        T temp = 0.0f;
        for (auto i = 0; i < row; i++) {
            temp += fabsf(pData[ i + row * j ]);
        }
        if (temp > norm)
            norm = temp;
    }

    return norm;
}


template <typename T>
T
NormMARS(DataType &aInput) {
    T norm = 0.0f;
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();

    auto pData = (T *) aInput.GetData();
    auto pTemp = new T[row];
    memset(pTemp, 0.0f, sizeof(T) * row);


    for (auto j = 0; j < col; j++) {
        for (auto i = 0; i < row; i++) {
            pTemp[ i ] += fabsf(pData[ i + row * j ]);
        }
    }

    for (auto i = 0; i < row; i++) {
        if (pTemp[ i ] > norm) {
            norm = pTemp[ i ];
        }
    }

    delete[] pTemp;
    return norm;
}


template <typename T>
T
NormEuclidean(DataType &aInput) {

    auto pData = (T *) aInput.GetData();
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();
    T scale = 0.0f;
    T sumsq = 1.0f;

    for (auto j = 0; j < col; j++) {
        lapack::lassq(row, pData + ( j * row ), 1, &scale, &sumsq);
    }

    return scale * sqrtf(sumsq);
}


template <typename T>
T
NormMaxMod(DataType &aInput) {

    auto pData = (T *) aInput.GetData();
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();
    T norm = 0.0f;

    for (auto j = 0; j < col; j++) {
        for (auto i = 0; i < row; i++) {
            T temp = (T) fabsf(pData[ i + row * j ]);
            if (temp > norm)
                norm = temp;
        }
    }

    return norm;
}

template <typename T>
void GetRank(DataType &aInput, const double aTolerance, T &aRank) {

    auto min_val = fabsf((T) aTolerance * aInput.GetVal(0));
    auto row = aInput.GetNRow();
    auto col = aInput.GetNCol();
    auto min_dim = std::min(row, col);

    for (auto i = 1; i < min_dim; i++) {
        if (fabsf((T) aInput.GetVal(i + row * i)) < min_val) {
            aRank = i;
            return;
        }
    }

    aRank = min_dim;
}


#endif //MPCR_LINEARALGEBRAHELPER_HPP
