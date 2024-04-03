/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <operations/concrete/GPUHelpers.hpp>
#include <utilities/MPCRDispatcher.hpp>
#include <utilities/TypeChecker.hpp>


using namespace mpcr::operations::helpers;
using namespace mpcr;


template <typename T>
__global__
void
SymmetrizeKernel(T *apData, size_t aSideLength, bool aToUpperTriangle) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < aSideLength && col < aSideLength) {
        // Calculate the linear indices for the lower and upper triangles
        size_t lowerIndex = col * aSideLength + row;  // Access lower triangle
        size_t upperIndex = row * aSideLength + col;  // Access upper triangle

        if (aToUpperTriangle) {
            if (row > col) {
                apData[ upperIndex ] = apData[ lowerIndex ];
            }
        } else {
            if (col > row) {
                apData[ upperIndex ] = apData[ lowerIndex ];
            }
        }
    }
}


template <typename T>
__global__ void
ReverseMatrixKernel(T *apData, size_t aNumRows, size_t aNumCol) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure thread indices are within matrix dimensions
    if (row < aNumRows && col < aNumCol / 2) {
        // Calculate indices for the element and its corresponding element in the reversed column
        size_t index = row + col * aNumRows;
        size_t reverseIndex = row + ( aNumCol - col - 1 ) * aNumRows;

        // Swap elements in the current column with their corresponding elements in the reversed column
        T temp = apData[ index ];
        apData[ index ] = apData[ reverseIndex ];
        apData[ reverseIndex ] = temp;
    }
}


template <typename T>
__global__
void ReverseArrayKernel(T *apData, size_t aSize) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < aSize / 2) {
        T temp = apData[ tid ];
        apData[ tid ] = apData[ aSize - tid - 1 ];
        apData[ aSize - tid - 1 ] = temp;
    }
}


template <typename T>
__global__
void
FillTriangleKernel(T *apData, size_t aSideLength, T aValue,
                   bool aUpperTriangle) {

    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;


    if (row < aSideLength && col < aSideLength) {
        size_t index = col * aSideLength + row;

        if (aUpperTriangle && col > row) {
            apData[ index ] = aValue;
        } else if (!aUpperTriangle && row > col) {
            apData[ index ] = aValue;
        }

    }
}


template <typename T>
__global__
void
TransposeKernel(T *apInput, T *apOutput, size_t aNumRow, size_t aNumCol) {

    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Load data from global memory into shared memory
    if (row < aNumRow && col < aNumCol) {
        apOutput[ row * aNumCol + col ] = apInput[ col * aNumRow + row ];
    }

}


template <typename T>
__global__
void
IdentityMatrixKernel(T *apData, size_t aSideLength) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < aSideLength && col < aSideLength) {
        apData[ row * aSideLength + col ] = ( row == col ) ? 1.0 : 0.0;
    }
}


template <typename T>
__global__
void
MACSKernel(T *apData, size_t aNumRow, size_t aNumCol, T *aOutput) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    T temp_val = 0;
    if (col < aNumCol) {
        for (auto i = 0; i < aNumRow; i++) {
            temp_val += fabsf(apData[ i + aNumRow * col ]);
        }
        aOutput[ col ] = temp_val;
    }
}


template <typename T>
__global__
void
MARSKernel(T *apData, size_t aNumRow, size_t aNumCol, T *aOutput) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    T temp_val = 0;
    if (col < aNumCol) {
        for (auto i = 0; i < aNumRow; i++) {
            temp_val += fabsf(apData[ i * aNumRow + col ]);
        }
        aOutput[ col ] = temp_val;
    }
}


template <typename T>
__global__ void
GetRankKernel(T *apData, size_t aNumRow, size_t aNumCol, int *apRank) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row == col && row < aNumRow && col < aNumCol) {
        if (apData[ row + aNumRow * col ] != 0) {
            atomicAdd((int *) apRank, 1);
        }
    }
}


/** -----------------------------  Drivers --------------------------------- **/
template <typename T>
void
GPUHelpers <T>::FillTriangle(DataType &aInput, const double &aValue,
                             const bool &aUpperTriangle,
                             kernels::RunContext *aContext) {

    if (aInput.GetNRow() != aInput.GetNCol()) {
        MPCR_API_EXCEPTION("Cannot Fill Square Matrix, Matrix is Not Square",
                           -1);
    }

    auto row = aInput.GetNRow();
    auto pData = (T *) aInput.GetData(GPU);

    auto side_len = aInput.GetNRow();


    dim3 block_size(MPCR_CUDA_BLOCK_SIZE, MPCR_CUDA_BLOCK_SIZE);
    dim3 grid_size(( side_len + block_size.x - 1 ) / block_size.x,
                   ( side_len + block_size.y - 1 ) / block_size.y);


    FillTriangleKernel <T><<<grid_size, block_size, 0, aContext->GetStream()>>>(
        pData, row, aValue, aUpperTriangle);

    aContext->Sync();

    aInput.SetData((char *) pData, GPU);

}


template <typename T>
void
GPUHelpers <T>::Transpose(DataType &aInput, kernels::RunContext *aContext) {
    auto row = aInput.GetNRow();
    auto col = aInput.GetNCol();
    auto pData = (T *) aInput.GetData(GPU);

    auto pData_transposed = (T *) memory::AllocateArray(row * col * sizeof(T),
                                                        GPU, aContext);
    aContext->Sync();


    dim3 block_size(MPCR_CUDA_BLOCK_SIZE, MPCR_CUDA_BLOCK_SIZE);
    dim3 grid_size(( col + block_size.x - 1 ) / block_size.x,
                   ( row + block_size.y - 1 ) / block_size.y);

    TransposeKernel <T><<<grid_size, block_size, 0, aContext->GetStream()>>>(
        pData, pData_transposed, row, col);


    aContext->Sync();
    aInput.SetData((char *) pData_transposed, GPU);
    aInput.SetDimensions(col, row);

}


template <typename T>
void
GPUHelpers <T>::Reverse(DataType &aInput, kernels::RunContext *aContext) {
    auto row = aInput.GetNRow();
    auto col = aInput.GetNCol();
    auto pData = (T *) aInput.GetData(GPU);
    auto num_elements = aInput.GetSize();
    auto is_matrix = aInput.IsMatrix();


    if (is_matrix) {
        dim3 block_size(MPCR_CUDA_BLOCK_SIZE, MPCR_CUDA_BLOCK_SIZE);
        dim3 grid_size(( col + block_size.x - 1 ) / block_size.x,
                       ( row + block_size.y - 1 ) / block_size.y);

        ReverseMatrixKernel <T><<<grid_size, block_size, 0, aContext->GetStream()>>>(
            pData, row, col);
    } else {

        auto threadsPerBlock = 256;
        auto blocksPerGrid =
            ( num_elements + threadsPerBlock - 1 ) / threadsPerBlock;

        ReverseArrayKernel <T><<<blocksPerGrid, threadsPerBlock, 0, aContext->GetStream()>>>(
            pData, num_elements);
    }

    aContext->Sync();
    aInput.SetData((char *) pData, GPU);

}


template <typename T>
void
GPUHelpers <T>::Symmetrize(DataType &aInput, const bool &aToUpperTriangle,
                           kernels::RunContext *aContext) {
    if (aInput.GetNRow() != aInput.GetNCol()) {
        MPCR_API_EXCEPTION("Cannot Symmetrize ,Matrix is Not Square", -1);
    }
    auto pData = (T *) aInput.GetData(GPU);
    auto side_len = aInput.GetNRow();


    dim3 block_size(MPCR_CUDA_BLOCK_SIZE, MPCR_CUDA_BLOCK_SIZE);
    dim3 grid_size(( side_len + block_size.x - 1 ) / block_size.x,
                   ( side_len + block_size.y - 1 ) / block_size.y);


    SymmetrizeKernel <T><<<grid_size,
    block_size, 0, aContext->GetStream()>>>
        (pData, side_len, aToUpperTriangle);


    aContext->Sync();
    aInput.SetData((char *) pData, GPU);


}


template <typename T>
void
GPUHelpers <T>::CreateIdentityMatrix(T *apData, size_t &aSideLength,
                                     kernels::RunContext *aContext) {

    dim3 block_size(MPCR_CUDA_BLOCK_SIZE, MPCR_CUDA_BLOCK_SIZE);
    dim3 grid_size(( aSideLength + block_size.x - 1 ) / block_size.x,
                   ( aSideLength + block_size.y - 1 ) / block_size.y);


    IdentityMatrixKernel <T><<<grid_size,
    block_size, 0, aContext->GetStream()>>>
        (apData, aSideLength);

    aContext->Sync();
}


template <typename T>
void
GPUHelpers <T>::NormMARS(DataType &aInput, T &aValue,
                         kernels::RunContext *aContext) {

    auto row = aInput.GetNRow();
    auto col = aInput.GetNCol();
    auto pData = (T *) aInput.GetData(GPU);


    auto shared_output = (T *) memory::AllocateArray(row * sizeof(T), GPU,
                                                     aContext);


    auto threadsPerBlock = 256;
    auto blocksPerGrid =
        ( col + threadsPerBlock - 1 ) / threadsPerBlock;

    MARSKernel <T><<<blocksPerGrid,
    threadsPerBlock, 0, aContext->GetStream()>>>
        (pData, row, col, shared_output);


    aContext->Sync();
    auto handle = aContext->GetCuBlasDnHandle();
    int idx = 0;
    if constexpr(is_double <T>()) {
        cublasIdamax(handle, row, shared_output, 1, &idx);
    } else {
        cublasIsamax(handle, row, shared_output, 1, &idx);
    }

    memory::MemCpy((char *) ( &aValue ), (char *) shared_output +
                                         ( idx * sizeof(T)), sizeof(T),
                   aContext,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    if (aValue < 0) {
        aValue = 0;
    }

    memory::DestroyArray((char *&) shared_output, GPU, aContext);
}


template <typename T>
void
GPUHelpers <T>::NormMACS(DataType &aInput, T &aValue,
                         kernels::RunContext *aContext) {
    auto row = aInput.GetNRow();
    auto col = aInput.GetNCol();
    auto pData = (T *) aInput.GetData(GPU);


    auto shared_output = (T *) memory::AllocateArray(col * sizeof(T), GPU,
                                                     aContext);


    auto threadsPerBlock = 256;
    auto blocksPerGrid =
        ( col + threadsPerBlock - 1 ) / threadsPerBlock;

    MARSKernel <T><<<blocksPerGrid,
    threadsPerBlock, 0, aContext->GetStream()>>>
        (pData, row, col, shared_output);


    aContext->Sync();
    auto handle = aContext->GetCuBlasDnHandle();
    int idx = 0;
    if constexpr(is_double <T>()) {
        cublasIdamax(handle, col, shared_output, 1, &idx);
    } else {
        cublasIsamax(handle, col, shared_output, 1, &idx);
    }

    memory::MemCpy((char *) ( &aValue ), (char *) shared_output +
                                         ( idx * sizeof(T)), sizeof(T),
                   aContext,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    if (aValue < 0) {
        aValue = 0;
    }


    memory::DestroyArray((char *&) shared_output, GPU, aContext);
}


template <typename T>
void
GPUHelpers <T>::NormEuclidean(DataType &aInput, T &aValue,
                              kernels::RunContext *aContext) {

    auto size = aInput.GetSize();
    auto pData = (T *) aInput.GetData(GPU);

    auto handle = aContext->GetCuBlasDnHandle();

    if constexpr(is_double <T>()) {
        cublasDnrm2(handle, size, pData, 1, &aValue);
    } else {
        cublasSnrm2(handle, size, pData, 1, &aValue);
    }


}


template <typename T>
void
GPUHelpers <T>::NormMaxMod(DataType &aInput, T &aValue,
                           kernels::RunContext *aContext) {
    auto size = aInput.GetSize();
    auto pData = (T *) aInput.GetData(GPU);

    auto handle = aContext->GetCuBlasDnHandle();
    int idx = 0;
    if constexpr(is_double <T>()) {
        cublasIdamax(handle, size, pData, 1, &idx);
    } else {
        cublasIsamax(handle, size, pData, 1, &idx);
    }

    memory::MemCpy((char *) ( &aValue ), (char *) pData +
                                         ( idx * sizeof(T)), sizeof(T),
                   aContext,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    if (aValue < 0) {
        aValue = 0;
    }
}


template <typename T>
void
GPUHelpers <T>::GetRank(DataType &aInput, const double &aTolerance, T &aRank,
                        kernels::RunContext *aContext) {

    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();

    auto pData = (T *) aInput.GetData(GPU);

    dim3 block_size(MPCR_CUDA_BLOCK_SIZE, MPCR_CUDA_BLOCK_SIZE);
    dim3 grid_size(( col + block_size.x - 1 ) / block_size.x,
                   ( row + block_size.y - 1 ) / block_size.y);

    auto rank = (int *) memory::AllocateArray(1 * sizeof(int), GPU, aContext);
    GetRankKernel <T><<<grid_size,
    block_size, 0, aContext->GetStream()>>>(pData, row, col, rank);


}