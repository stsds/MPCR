/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <operations/concrete/CPUHelpers.hpp>
#include <utilities/MPCRDispatcher.hpp>
#include <lapack.hh>


using namespace mpcr::operations::helpers;
using namespace mpcr;


template <typename T>
void
ReverseMatrix(DataType &aInput) {
    auto pData = (T *) aInput.GetData(CPU);
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();

    auto last = col - 1;

    for (auto j = 0; j < col / 2; j++) {
        for (auto i = 0; i < row; i++) {
            std::swap(pData[ i + row * j ], pData[ i + row * last ]);
        }
        last--;
    }

    aInput.SetData((char *) pData, CPU);
}


template <typename T>
void
ReverseVector(DataType &aInput) {
    auto pData = (T *) aInput.GetData(CPU);
    auto size = aInput.GetSize();
    std::reverse(pData, pData + size);
    aInput.SetData((char *) pData, CPU);
}


template <typename T>
void
CPUHelpers <T>::FillTriangle(DataType &aInput, const double &aValue,
                             const bool &aUpperTriangle,
                             kernels::RunContext *aContext) {

    if (aInput.GetNRow() != aInput.GetNCol()) {
        MPCR_API_EXCEPTION("Cannot Fill Square Matrix, Matrix is Not Square",
                           -1);
    }

    aInput.FillTriangle(aValue, aUpperTriangle);
}


template <typename T>
void
CPUHelpers <T>::Transpose(DataType &aInput, kernels::RunContext *aContext) {
    aInput.Transpose();
}


template <typename T>
void
CPUHelpers <T>::Reverse(DataType &aInput, kernels::RunContext *aContext) {

    auto is_matrix = aInput.IsMatrix();

    if (is_matrix) {
        ReverseMatrix <T>(aInput);
    } else {
        ReverseVector <T>(aInput);
    }


}


template <typename T>
void
CPUHelpers <T>::Symmetrize(DataType &aInput, const bool &aToUpperTriangle,
                           kernels::RunContext *aContext) {

    if (aInput.GetNRow() != aInput.GetNCol()) {
        MPCR_API_EXCEPTION("Cannot Symmetrize ,Matrix is Not Square", -1);
    }

    auto pData = (T *) aInput.GetData(CPU);
    auto row = aInput.GetNRow();
    auto col = aInput.GetNCol();

    if (aToUpperTriangle) {

        for (auto j = 0; j < row; j += MPCR_CPU_BLOCK_SIZE) {
            for (auto i = j + 1; i < row; i += MPCR_CPU_BLOCK_SIZE) {
                for (auto col_idx = j;
                     col_idx < j + MPCR_CPU_BLOCK_SIZE &&
                     col_idx < row; ++col_idx) {
                    for (auto row_idx = i;
                         row_idx < i + MPCR_CPU_BLOCK_SIZE &&
                         row_idx < row; ++row_idx)
                        pData[ col_idx + row * row_idx ] = pData[ row_idx +
                                                                  row *
                                                                  col_idx ];
                }
            }
        }

    } else {

        for (auto j = 0; j < row; j += MPCR_CPU_BLOCK_SIZE) {
            for (auto i = j + 1; i < row; i += MPCR_CPU_BLOCK_SIZE) {
                for (auto col_idx = j;
                     col_idx < j + MPCR_CPU_BLOCK_SIZE &&
                     col_idx < row; ++col_idx) {
                    for (auto row_idx = i;
                         row_idx < i + MPCR_CPU_BLOCK_SIZE &&
                         row_idx < row; ++row_idx)
                        pData[ row_idx + row * col_idx ] = pData[ col_idx +
                                                                  row *
                                                                  row_idx ];
                }
            }
        }

    }

    aInput.SetData((char *) pData, CPU);
}


template <typename T>
void
CPUHelpers <T>::NormMARS(DataType &aInput, T &aValue) {

    aValue = 0.0f;
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();

    auto pData = (T *) aInput.GetData(CPU);
    auto pTemp = (T *) memory::AllocateArray(row * sizeof(T), CPU, nullptr);
    memory::Memset((char *) pTemp, 0, sizeof(T) * row, CPU, nullptr);


    for (auto j = 0; j < col; j++) {
        for (auto i = 0; i < row; i++) {
            pTemp[ i ] += fabsf(pData[ i + row * j ]);
        }
    }

    for (auto i = 0; i < row; i++) {
        if (pTemp[ i ] > aValue) {
            aValue = pTemp[ i ];
        }
    }

    memory::DestroyArray((char *&) pTemp, CPU, nullptr);

}


template <typename T>
void
CPUHelpers <T>::NormMACS(DataType &aInput, T &aValue) {

    aValue = 0.0f;
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();
    auto pData = (T *) aInput.GetData(CPU);

    for (auto j = 0; j < col; j++) {
        T temp = 0.0f;
        for (auto i = 0; i < row; i++) {
            temp += fabsf(pData[ i + row * j ]);
        }

        if (temp > aValue)
            aValue = temp;
    }

}


template <typename T>
void
CPUHelpers <T>::NormEuclidean(DataType &aInput, T &aValue) {

    auto pData = (T *) aInput.GetData(CPU);
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();
    T scale = 0.0f;
    T sumsq = 1.0f;

    for (auto j = 0; j < col; j++) {
        lapack::lassq(row, pData + ( j * row ), 1, &scale, &sumsq);
    }

    aValue = scale * sqrtf(sumsq);

}

template <typename T>
void
CPUHelpers <T>::NormMaxMod(DataType &aInput, T &aValue) {

    auto pData = (T *) aInput.GetData(CPU);
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();
    aValue = 0.0f;

    for (auto j = 0; j < col; j++) {
        for (auto i = 0; i < row; i++) {
            T temp = (T) fabsf(pData[ i + row * j ]);
            if (temp > aValue)
                aValue = temp;
        }
    }

}

template <typename T>
void
CPUHelpers <T>::GetRank(DataType &aInput, const double &aTolerance, T &aRank) {
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


template <typename T>
void
CPUHelpers <T>::CreateIdentityMatrix(T *apData, size_t &aSideLength,
                                     kernels::RunContext *aContext) {
    MPCR_API_EXCEPTION("CPU Identity Matrix is not implemented", -1);
}