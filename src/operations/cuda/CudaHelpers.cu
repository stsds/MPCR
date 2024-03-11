

#include <operations/cuda/CudaHelpers.hpp>
#include <utilities/MPCRDispatcher.hpp>


using namespace mpcr::operations::helpers;
using namespace mpcr;


template <typename T>
__global__
void
SymmetrizeKernel(T *aData, size_t aSideLength, bool aToUpperTriangle) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < aSideLength && col < aSideLength) {
        // Calculate the linear indices for the lower and upper triangles
        size_t lowerIndex = col * aSideLength + row;  // Access lower triangle
        size_t upperIndex = row * aSideLength + col;  // Access upper triangle

        if (aToUpperTriangle) {
            if (row > col) {
                aData[ upperIndex ] = aData[ lowerIndex ];
            }
        } else {
            if (col > row) {
                aData[ upperIndex ] = aData[ lowerIndex ];
            }
        }
    }
}


template <typename T>
__global__ void
ReverseMatrixKernel(T *apData,size_t aNumRows, size_t aNumCol) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure thread indices are within matrix dimensions
    if (row < aNumRows && col < aNumCol / 2) {
        // Calculate indices for the element and its corresponding element in the reversed column
        size_t index = row + col * aNumRows;
        size_t reverseIndex = row + (aNumCol - col - 1) * aNumRows;

        // Swap elements in the current column with their corresponding elements in the reversed column
        T temp = apData[index];
        apData[index] = apData[reverseIndex];
        apData[reverseIndex] = temp;
    }
}


template <typename T>
__global__
void ReverseArrayKernel(T *apData,size_t aSize) {
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


/** -----------------------------  Drivers --------------------------------- **/
template <typename T>
void
CudaHelpers::FillTriangle(DataType &aInput, const double &aValue,
                          const bool &aUpperTriangle,
                          kernels::RunContext *aContext) {

    if (aInput.GetNRow() != aInput.GetNCol()) {
        MPCR_API_EXCEPTION("Cannot Fill Square Matrix, Matrix is Not Square",
                           -1);
    }

    auto row = aInput.GetNRow();
    auto pData = (T *) aInput.GetData(GPU);

    auto side_len = aInput.GetNRow();


    dim3 blockSize(16, 16);
    dim3 gridSize(( side_len + blockSize.x - 1 ) / blockSize.x,
                  ( side_len + blockSize.y - 1 ) / blockSize.y);


    FillTriangleKernel <T><<<gridSize, blockSize, 0, aContext->GetStream()>>>(
        pData, row, aValue, aUpperTriangle);

    aContext->Sync();

    aInput.SetData((char *) pData, GPU);

}


template <typename T>
void
CudaHelpers::Reverse(DataType &aInput, kernels::RunContext *aContext) {
    auto row = aInput.GetNRow();
    auto col = aInput.GetNCol();
    auto pData = (T *) aInput.GetData(GPU);
    auto num_elements = aInput.GetSize();
    auto is_matrix = aInput.IsMatrix();


    if (is_matrix) {
        dim3 blockSize(16, 16);
        dim3 gridSize(( col + blockSize.x - 1 ) / blockSize.x,
                      ( row + blockSize.y - 1 ) / blockSize.y);

        ReverseMatrixKernel <T><<<gridSize, blockSize, 0, aContext->GetStream()>>>(
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
CudaHelpers::Symmetrize(DataType &aInput, const bool &aToUpperTriangle,
                        kernels::RunContext *aContext) {
    if (aInput.GetNRow() != aInput.GetNCol()) {
        MPCR_API_EXCEPTION("Cannot Symmetrize ,Matrix is Not Square", -1);
    }
    auto pData = (T *) aInput.GetData(GPU);
    auto side_len = aInput.GetNRow();


    dim3 blockSize(16, 16);
    dim3 gridSize(( side_len + blockSize.x - 1 ) / blockSize.x,
                  ( side_len + blockSize.y - 1 ) / blockSize.y);


    SymmetrizeKernel <T><<<gridSize,
    blockSize, 0, aContext->GetStream()>>>
        (pData, side_len, aToUpperTriangle);


    aContext->Sync();
    aInput.SetData((char *) pData, GPU);


}


SIMPLE_INSTANTIATE(void, CudaHelpers::Symmetrize, DataType &aInput,
                   const bool &aToUpperTriangle, kernels::RunContext *aContext)


SIMPLE_INSTANTIATE(void, CudaHelpers::Reverse, DataType &aInput,
                   kernels::RunContext *aContext)

SIMPLE_INSTANTIATE(void, CudaHelpers::FillTriangle, DataType &aInput,
                   const double &aValue, const bool &aUpperTriangle,
                   kernels::RunContext *aContext)
