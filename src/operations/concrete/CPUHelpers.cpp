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
CPUHelpers <T>::NormMARS(DataType &aInput, T &aValue,
                         kernels::RunContext *aContext) {

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
CPUHelpers <T>::NormMACS(DataType &aInput, T &aValue,
                         kernels::RunContext *aContext) {

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
CPUHelpers <T>::NormEuclidean(DataType &aInput, T &aValue,
                              kernels::RunContext *aContext) {

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
CPUHelpers <T>::NormMaxMod(DataType &aInput, T &aValue,
                           kernels::RunContext *aContext) {

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
CPUHelpers <T>::GetRank(DataType &aInput, T &aRank,
                        kernels::RunContext *aContext) {
    auto row = aInput.GetNRow();
    auto col = aInput.GetNCol();
    auto min_dim = std::min(row, col);
    aRank = 0;

    for (auto i = 1; i < min_dim; i++) {
        if (fabsf((T) aInput.GetVal(i + row * i)) != 0) {
            aRank += 1;
        }
    }
}


template <typename T>
void
CPUHelpers <T>::IsSymmetric(DataType &aInput, bool &aOutput,
                            kernels::RunContext *aContext) {

    aOutput = false;
    auto pData = (T *) aInput.GetData(CPU);
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();

    if (col != row) {
        return;
    }

    size_t idx_col_maj;
    size_t idx_row_maj;
    auto epsilon = std::numeric_limits <T>::epsilon();
    T val;
    for (auto i = 0; i < col; i++) {
        for (auto j = 0; j < row; j++) {
            if (i == j) {
                break;
            }
            idx_col_maj = ( i * row ) + j;
            idx_row_maj = ( j * col ) + i;
            val = std::fabs(pData[ idx_row_maj ] - pData[ idx_col_maj ]);
            if (val > epsilon) {
                return;
            }
        }
    }

    aOutput = true;
}




template <typename T>
void
CPUHelpers <T>::CreateIdentityMatrix(T *apData, size_t &aSideLength,
                                     kernels::RunContext *aContext) {
    MPCR_API_EXCEPTION("CPU Identity Matrix is not implemented", -1);
}


template <typename T>
void
CPUHelpers <T>::CopyUpperTriangle(DataType &aInput, DataType &aOutput,
                                  kernels::RunContext *aContext) {

    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();
    auto output_nrows = aOutput.GetNRow();
    auto output_size=aOutput.GetSize();


    auto pData_src = (T *) aInput.GetData(CPU);
    auto pData_dest = (T *) aOutput.GetData(CPU);

    memset(pData_dest, 0, output_size * sizeof(T));

    for (auto j = 0; j < col; j++) {
        for (auto i = 0; i <= j && i < output_nrows; i++){
            pData_dest[ i + output_nrows * j ] = pData_src[ i + row * j ];
        }
    }

    aOutput.SetData((char *) pData_dest, CPU);

}