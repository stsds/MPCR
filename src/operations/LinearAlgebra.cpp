/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <operations/helpers/LinearAlgebraHelper.hpp>
#include <operations/LinearAlgebra.hpp>
#include <utilities/TypeChecker.hpp>
#include <operations/concrete/LinearAlgebraBackendFactory.hpp>


#ifdef USE_CUDA

#include <operations/cuda/CudaHelpers.hpp>


#endif

using namespace mpcr::operations;
using namespace mpcr::kernels;
using namespace std;


template <typename T>
void
linear::CrossProduct(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                     const bool &aTransposeA, const bool &aTransposeB,
                     const bool &aSymmetrize, const double &aAlpha,
                     const double &aBeta) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    auto is_one_input = aInputB.GetSize() == 0;
    auto flag_conv = false;

    if (!aInputB.IsMatrix() && !is_one_input) {
        if (aInputA.IsMatrix()) {
            if (aInputA.GetNCol() == aInputB.GetNCol()) {
                aInputB.SetDimensions(aInputA.GetNCol(), 1);
                flag_conv = true;
            }
        }
    }

    if (!aInputA.IsMatrix() && !is_one_input) {
        if (aInputB.IsMatrix()) {
            if (aInputA.GetNCol() != aInputB.GetNRow()) {
                aInputA.SetDimensions(aInputA.GetNCol(), 1);
                flag_conv = true;
            }
        }
    }

    auto row_a = aInputA.GetNRow();
    auto col_a = aInputA.GetNCol();

    size_t row_b;
    size_t col_b;

    // cross(x,y) -> x y
    // tcross(x,y) -> x t(y)

    // cross(x) -> t(x) x
    // tcross(x) -> x t(x)

    if (is_one_input) {
        row_b = row_a;
        col_b = col_a;
    } else {
        row_b = aInputB.GetNRow();
        col_b = aInputB.GetNCol();
    }

    size_t lda = row_a;
    size_t ldb = row_b;

    if (aTransposeA) {
        std::swap(row_a, col_a);
    }
    if (aTransposeB) {
        std::swap(row_b, col_b);
    }

    if (col_a != row_b) {
        MPCR_API_EXCEPTION("Wrong Matrix Dimensions", -1);
    }


    T *pData_out = nullptr;

    if (aOutput.GetSize() != 0) {

        if (aOutput.GetNRow() != row_a || aOutput.GetNCol() != col_b) {
            MPCR_API_EXCEPTION("Wrong Output Matrix Dimensions", -1);
        }

        pData_out = (T *) aOutput.GetData(operation_placement);

    } else {
        auto output_size = row_a * col_b;
        pData_out = (T *) memory::AllocateArray(output_size * sizeof(T),
                                                operation_placement, context);
        memory::Memset((char *) pData_out, 0, sizeof(T) * output_size,
                       operation_placement, context);

        aOutput.ClearUp();
        aOutput.SetSize(output_size);
        aOutput.SetDimensions(row_a, col_b);
    }

    auto pData_a = (T *) aInputA.GetData(operation_placement);
    auto pData_b = (T *) aInputB.GetData(operation_placement);

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);

    if (!is_one_input) {
        solver->Gemm(aTransposeA, aTransposeB, row_a, col_b, col_a, aAlpha,
                     pData_a, lda, pData_b, ldb, aBeta, pData_out, row_a);
    } else {
        solver->Syrk(true, aTransposeA, row_a, col_a, aAlpha, pData_a, lda,
                     aBeta, pData_out, row_a);

    }

    aOutput.SetData((char *) pData_out, operation_placement);

    if (is_one_input && aSymmetrize) {
        if (operation_placement == CPU) {
            Symmetrize <T>(aOutput, true);
        } else {
            helpers::CudaHelpers::Symmetrize <T>(aOutput, true, context);
        }
    }

    if (flag_conv) {
        aInputB.ToVector();
    }

}


template <typename T>
void
linear::IsSymmetric(DataType &aInput, bool &aOutput) {

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
linear::Cholesky(DataType &aInputA, DataType &aOutput,
                 const bool &aUpperTriangle) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    auto row = aInputA.GetNRow();
    auto col = aInputA.GetNCol();


    if (row != col) {
        MPCR_API_EXCEPTION(
            "Cannot Apply Cholesky Decomposition on non-square Matrix", -1);
    }

    auto pData = (T *) aInputA.GetData(operation_placement);
    auto pOutput = memory::AllocateArray(row * col * sizeof(T),
                                         operation_placement, context);

    auto mem_transfer = ( operation_placement == CPU )
                        ? memory::MemoryTransfer::HOST_TO_HOST
                        : memory::MemoryTransfer::DEVICE_TO_DEVICE;

    memory::MemCpy(pOutput, (char *) pData, row * col * sizeof(T), context,
                   mem_transfer);


    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);

    auto rc = solver->Potrf(aUpperTriangle, row, (T *) pOutput, row);

    if (rc != 0) {
        MPCR_API_EXCEPTION(
            "Error While Applying Cholesky Decomposition", rc);
    }

    aOutput.ClearUp();
    aOutput.SetDimensions(aInputA);
    aOutput.SetData((char *) pOutput, operation_placement);
    if (operation_placement == CPU) {
        aOutput.FillTriangle(0, !aUpperTriangle);
    } else {
        helpers::CudaHelpers::FillTriangle <T>(aOutput, 0, !aUpperTriangle,
                                               context);
    }

}


template <typename T>
void
linear::CholeskyInv(DataType &aInputA, DataType &aOutput, const size_t &aNCol) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    auto col = aInputA.GetNCol();

    if (aNCol > col) {
        MPCR_API_EXCEPTION(
            "Size Cannot exceed the Number of Cols of Input", -1);
    }

    T *pOutput = nullptr;


    aOutput.ClearUp();
    if (aNCol == col) {
        aOutput.SetSize(aNCol * aNCol);
        aOutput.SetDimensions(aNCol, aNCol);
        auto pData = (T *) aInputA.GetData(operation_placement);

        pOutput = (T *) memory::AllocateArray(aNCol * aNCol * sizeof(T),
                                              operation_placement, context);
        auto mem_transfer = ( operation_placement == CPU )
                            ? memory::MemoryTransfer::HOST_TO_HOST
                            : memory::MemoryTransfer::DEVICE_TO_DEVICE;

        memory::MemCpy((char *) pOutput, (char *) pData,
                       aNCol * aNCol * sizeof(T), context,
                       mem_transfer);

    } else {
        auto pData = (T *) aInputA.GetData(CPU);
        auto new_size = aNCol * aNCol;
        aOutput.SetSize(new_size);
        aOutput.SetDimensions(aNCol, aNCol);
        auto pTemp_data = (T *) memory::AllocateArray(new_size * sizeof(T), CPU,
                                                      nullptr);

        /** TODO: for better optimization, this kernel should be implemented for GPU **/
        size_t idx;
        for (auto i = 0; i < aNCol; i++) {
            for (auto j = 0; j < aNCol; j++) {
                idx = j + ( i * aNCol );
                pTemp_data[ idx ] = pData[ j + ( col * i ) ];
            }
        }
        if (operation_placement == CPU) {
            pOutput = pTemp_data;
        } else {
            pOutput = (T *) memory::AllocateArray(new_size * sizeof(T), GPU,
                                                  context);

            memory::MemCpy((char *) pOutput, (char *) pTemp_data,
                           new_size * sizeof(T), context,
                           memory::MemoryTransfer::HOST_TO_DEVICE);
        }
    }


    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);
    auto rc = solver->Potri(true, aNCol, pOutput, aOutput.GetNRow());

    if (rc != 0) {
        MPCR_API_EXCEPTION(
            "Error While Applying Cholesky Decomposition", rc);
    }


    aOutput.SetData((char *) pOutput, operation_placement);
    if (operation_placement == CPU) {
        Symmetrize <T>(aOutput, false);
    } else {
        helpers::CudaHelpers::Symmetrize <T>(aOutput, false, context);
    }

}


template <typename T>
void linear::Solve(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                   const bool &aSingle) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    auto rows_a = aInputA.GetNRow();
    auto cols_a = aInputA.GetNCol();
    bool flag_to_matrix = false;


    if (rows_a != cols_a) {
        MPCR_API_EXCEPTION("Cannot Solve This Matrix , Must be a Square Matrix",
                           -1);
    }

    auto rows_b = rows_a;
    auto cols_b = rows_b;

    if (!aSingle) {
        if (!aInputB.IsMatrix()) {
            flag_to_matrix = true;
            aInputB.SetDimensions(aInputB.GetNCol(), 1);
        }
        rows_b = aInputB.GetNRow();
        cols_b = aInputB.GetNCol();
    }

    if (cols_a != rows_b) {
        MPCR_API_EXCEPTION("Dimensions must be compatible", -1);
    }


    auto pIpiv = (int64_t *) memory::AllocateArray(cols_a * sizeof(int64_t),
                                                   operation_placement,
                                                   context);
    aOutput.ClearUp();
    auto rc = 0;

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);

    if (!aSingle) {
        DataType dump = aInputA;
        auto pData_dump = (T *) dump.GetData(operation_placement);
        T *pData_in_out = nullptr;

        /**
         * Incase of using CPU backend, lapack will overwrite the pData_in_out.
         * However, when using GPU, CuBlas will not change pData_in_out and instead
         * will change the output pointer.
         * **/
        if (operation_placement == definitions::CPU) {
            aOutput = aInputB;
            pData_in_out = (T *) aOutput.GetData(operation_placement);
            rc = solver->Gesv(cols_a, cols_b, pData_dump, rows_a,
                              (void *) pIpiv, nullptr, rows_b, pData_in_out,
                              rows_b);

        } else {
            aOutput.SetDimensions(aInputB);
            pData_in_out = (T *) memory::AllocateArray(
                aOutput.GetSize() * sizeof(T), GPU, context);

            rc = solver->Gesv(cols_a, cols_b, pData_dump, rows_a,
                              (void *) pIpiv, (T *) aInputB.GetData(), rows_b,
                              pData_in_out, rows_b);
        }
        aOutput.SetData((char *) pData_in_out, operation_placement);

    } else {
        aOutput = aInputA;
        auto pData_in_out = (T *) aOutput.GetData(operation_placement);

        rc = solver->Getrf(rows_a, cols_a, pData_in_out, rows_a, pIpiv);

        if (rc != 0) {
            delete[] pIpiv;
            MPCR_API_EXCEPTION("Error While Solving", rc);
        }

        rc = solver->Getri(cols_a, pData_in_out, rows_a, pIpiv);
        if (rc != 0) {
            delete[] pIpiv;
            MPCR_API_EXCEPTION("Error While Solving", rc);
        }
        aOutput.SetData((char *) pData_in_out, operation_placement);

    }

    if (rc != 0) {
        delete[] pIpiv;
        MPCR_API_EXCEPTION("Error While Solving", rc);
    }


    aOutput.SetSize(cols_a * cols_b);
    aOutput.SetDimensions(cols_a, cols_b);
    if (flag_to_matrix) {
        aInputB.ToVector();
    }

    delete[] pIpiv;
}


template <typename T>
void
linear::BackSolve(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                  const size_t &aCol, const bool &aUpperTri,
                  const bool &aTranspose, const char &aSide,
                  const double &aAlpha) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    bool flag_transform = false;
    if (!aInputA.IsMatrix()) {
        MPCR_API_EXCEPTION(
            "Inputs Must Be Matrices", -1);
    }

    if (!aInputB.IsMatrix()) {
        aInputB.SetDimensions(aInputB.GetNCol(), 1);
        flag_transform = true;

    }
    auto row_a = aInputA.GetNRow();
    auto row_b = aInputB.GetNRow();
    auto col_b = aInputB.GetNCol();
    auto left_side = aSide == 'L';

    if (aCol > row_a || std::isnan(aCol) || aCol < 1) {
        MPCR_API_EXCEPTION(
            "Given Number of Columns is Greater than Columns of B", -1);
    }

    aOutput.ClearUp();
    aOutput.SetSize(col_b * aCol);
    aOutput.SetDimensions(aCol, col_b);

    auto pData = (T *) aInputA.GetData(operation_placement);
    auto pData_b = (T *) aInputB.GetData(operation_placement);
    auto pData_in_out = (T *) memory::AllocateArray(col_b * aCol * sizeof(T),
                                                    operation_placement,
                                                    context);

    auto mem_transfer = ( operation_placement == CPU )
                        ? memory::MemoryTransfer::HOST_TO_HOST
                        : memory::MemoryTransfer::DEVICE_TO_DEVICE;

    for (auto i = 0; i < col_b; i++) {
        memory::MemCpy((char *) ( pData_in_out + ( aCol * i )),
                       (char *) ( pData_b + ( row_b * i )),
                       ( sizeof(T) * aCol ), context, mem_transfer);
    }


    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);

    solver->Trsm(left_side, aUpperTri, aTranspose, row_b, col_b, aAlpha, pData,
                 row_a, pData_in_out, row_b);


    aOutput.SetData((char *) pData_in_out, operation_placement);
    if (flag_transform) {
        aInputB.ToVector();
    }


}


template <typename T>
void
linear::SVD(DataType &aInputA, DataType &aOutputS, DataType &aOutputU,
            DataType &aOutputV, const size_t &aNu,
            const size_t &aNv, const bool &aTranspose) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    //s ,u ,vt
    auto row = aInputA.GetNRow();
    auto col = aInputA.GetNCol();
    auto pData = (T *) aInputA.GetData(operation_placement);

    auto min_dim = std::min(row, col);
    auto pOutput_s = (T *) memory::AllocateArray(min_dim * sizeof(T),
                                                 operation_placement, context);
    T *pOutput_u = nullptr;
    T *pOutput_vt = nullptr;

    aOutputS.ClearUp();
    aOutputU.ClearUp();
    aOutputV.ClearUp();

    aOutputS.SetSize(min_dim);

    if (aNu) {
        pOutput_u = (T *) memory::AllocateArray(row * aNu * sizeof(T),
                                                operation_placement, context);
        aOutputU.SetSize(row * aNu);
        aOutputU.SetDimensions(row, aNu);
    }

    if (aNv) {
        pOutput_vt = (T *) memory::AllocateArray(col * aNv * sizeof(T),
                                                 operation_placement, context);
        aOutputV.SetSize(col * aNv);
        /** Will be transposed at the end in case of svd **/
        aOutputV.SetDimensions(aNv, col);
    }


    auto pTemp_data = (T *) memory::AllocateArray(row * col * sizeof(T),
                                                  operation_placement, context);

    auto mem_transfer = ( operation_placement == CPU )
                        ? memory::MemoryTransfer::HOST_TO_HOST
                        : memory::MemoryTransfer::DEVICE_TO_DEVICE;

    memory::MemCpy((char *) pTemp_data, (char *) pData,
                   ( row * col ) * sizeof(T), context, mem_transfer);


    signed char job;
    int ldvt;
    if (aNu == 0 && aNv == 0) {
        job = 'N'; // NoVec
        ldvt = 1;
    } else if (aNu <= min_dim && aNv <= min_dim) {
        job = 'S'; // SomeVec
        ldvt = min_dim;
    } else {
        job = 'A'; //AllVec
        ldvt = aNv;
    }

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);

    // Gesdd routine in CPU
    auto rc = solver->SVD(job, row, col, pTemp_data, row, pOutput_s, pOutput_u,
                          row, pOutput_vt, ldvt);

    if (rc != 0) {
        delete[] pOutput_vt;
        delete[] pOutput_u;
        delete[] pOutput_s;
        delete[] pTemp_data;
        MPCR_API_EXCEPTION("Error While Getting SVD", rc);
    }


    aOutputS.SetData((char *) pOutput_s, operation_placement);
    aOutputV.SetData((char *) pOutput_vt, operation_placement);
    aOutputU.SetData((char *) pOutput_u, operation_placement);
    if (aTranspose) {
        if (operation_placement == CPU) {
            aOutputV.Transpose();
        } else {
            helpers::CudaHelpers::Transpose <T>(aOutputV, context);
        }
    }

}


template <typename T>
void linear::Eigen(DataType &aInput, DataType &aOutputValues,
                   DataType *apOutputVectors) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();

    if (row != col) {
        MPCR_API_EXCEPTION("Cannot Perform Eigen on non square Matrix", -1);
    }


    auto jobz_no_vec = true;
    auto fill_upper = true;

    if (apOutputVectors != nullptr) {
        jobz_no_vec = false;
    }

    auto pData = (T *) aInput.GetData(operation_placement);


    auto pValues = (T *) memory::AllocateArray(col * sizeof(T),
                                               operation_placement, context);

    auto pVectors = (T *) memory::AllocateArray(col * col * sizeof(T),
                                                operation_placement, context);

    auto mem_transfer = ( operation_placement == CPU )
                        ? memory::MemoryTransfer::HOST_TO_HOST
                        : memory::MemoryTransfer::DEVICE_TO_DEVICE;


    memory::MemCpy((char *) pVectors, (char *) pData, col * col * sizeof(T),
                   context, mem_transfer);

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);

    auto rc = solver->Syevd(jobz_no_vec, fill_upper, col, pVectors, col,
                            pValues);

    if (rc != 0) {
        delete[] pValues;
        delete[] pVectors;
        MPCR_API_EXCEPTION("Error While Performing Eigen", rc);
    }

    if (apOutputVectors) {
        apOutputVectors->ClearUp();
        apOutputVectors->SetSize(col * col);
        apOutputVectors->SetDimensions(col, col);
        apOutputVectors->SetData((char *) pVectors, operation_placement);
        if (operation_placement == CPU) {
            ReverseMatrix <T>(*apOutputVectors);
        } else {
            helpers::CudaHelpers::Reverse <T>(*apOutputVectors, context);
        }
    } else {
        delete[] pVectors;
    }

    if (operation_placement == CPU) {
        std::reverse(pValues, pValues + col);
    }
    aOutputValues.ClearUp();
    aOutputValues.SetSize(col);
    aOutputValues.SetData((char *) pValues, operation_placement);

    if (operation_placement == GPU) {
        helpers::CudaHelpers::Reverse <T>(aOutputValues, context);
    }


}


template <typename T>
void
linear::Norm(DataType &aInput, const std::string &aType, DataType &aOutput) {

    /** TODO: CPU implementation only **/
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();
    aOutput.ClearUp();
    aOutput.SetSize(1);

    auto pOutput = (T *) memory::AllocateArray(1 * sizeof(T), CPU, nullptr);

    if (row == 0 || col == 0) {
        pOutput[ 0 ] = 0.0f;
    } else if (aType == "O" || aType == "1") {
        pOutput[ 0 ] = NormMACS <T>(aInput);
    } else if (aType == "I") {
        pOutput[ 0 ] = NormMARS <T>(aInput);
    } else if (aType == "F") {
        pOutput[ 0 ] = NormEuclidean <T>(aInput);
    } else if (aType == "M") {
        pOutput[ 0 ] = NormMaxMod <T>(aInput);
    } else {
        delete[] pOutput;
        MPCR_API_EXCEPTION(
            "Argument must be one of 'M','1','O','I','F' or 'E' ",
            -1);
    }

    aOutput.SetData((char *) pOutput, CPU);
}


template <typename T>
void
linear::QRDecomposition(DataType &aInputA, DataType &aOutputQr,
                        DataType &aOutputQraux, DataType &aOutputPivot,
                        DataType &aRank, const double &aTolerance) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    auto mem_transfer = ( operation_placement == CPU )
                        ? memory::MemoryTransfer::HOST_TO_HOST
                        : memory::MemoryTransfer::DEVICE_TO_DEVICE;

    auto col = aInputA.GetNCol();
    auto row = aInputA.GetNRow();
    auto min_dim = std::min(col, row);
    auto pData = (T *) aInputA.GetData(operation_placement);

    auto pQr_in_out = (T *) memory::AllocateArray(row * col * sizeof(T),
                                                  operation_placement, context);
    auto pQraux = (T *) memory::AllocateArray(min_dim * sizeof(T),
                                              operation_placement, context);
    auto pJpvt = (int64_t *) memory::AllocateArray(col * sizeof(int64_t),
                                                   operation_placement,
                                                   context);

    memory::Memset((char *) pJpvt, 0, col * sizeof(int64_t),
                   operation_placement, context);


    memory::MemCpy((char *) pQr_in_out, (char *) pData,
                   ( aInputA.GetSize()) * sizeof(T), context, mem_transfer);

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);

    auto rc = solver->Geqp3(row, col, pQr_in_out, row, pJpvt, pQraux);

    if (rc != 0) {
        delete[] pQr_in_out;
        delete[] pJpvt;
        delete[] pQraux;
        MPCR_API_EXCEPTION("Error While Performing QR Decomposition", rc);
    }


    aOutputQr.ClearUp();
    aOutputPivot.ClearUp();
    aOutputQraux.ClearUp();

    aOutputQr.SetSize(row * col);
    aOutputQr.SetDimensions(row, col);
    aOutputQr.SetData((char *) pQr_in_out, operation_placement);

    aOutputQraux.SetSize(min_dim);
    aOutputQraux.SetData((char *) pQraux, operation_placement);

    auto pTemp_pvt = (T *) memory::AllocateArray(col * sizeof(T),
                                                 operation_placement, context);

    //TODO:: revise from here


    memory::Copy <int64_t, T>((char *) pJpvt, (char *) pTemp_pvt, col,
                              operation_placement);

    delete[] pJpvt;

    aOutputPivot.SetSize(col);
    aOutputPivot.SetData((char *) pTemp_pvt, operation_placement);

    //TODO: needs to be revisited in GPU
    auto pRank = (T *) memory::AllocateArray(1 * sizeof(T), CPU, nullptr);
    GetRank <T>(aOutputQr, aTolerance, *pRank);

    aRank.ClearUp();
    aRank.SetSize(1);
    aRank.SetData((char *) pRank, CPU);

}


template <typename T>
void
linear::QRDecompositionR(DataType &aInputA, DataType &aOutput,
                         const bool &aComplete) {

//TODO:: CPU Implementation only

    auto col = aInputA.GetNCol();
    auto row = aInputA.GetNRow();
    auto output_nrows = aComplete ? row : std::min(row, col);
    auto output_size = output_nrows * col;
    auto pOutput_data = (T *) memory::AllocateArray(output_size * sizeof(T),
                                                    CPU, nullptr);
    auto pData = (T *) aInputA.GetData();

    memset(pOutput_data, 0, output_size * sizeof(T));

    for (auto j = 0; j < col; j++) {
        for (auto i = 0; i <= j && i < output_nrows; i++)
            pOutput_data[ i + output_nrows * j ] = pData[ i + row * j ];
    }

    aOutput.ClearUp();
    aOutput.SetSize(output_size);
    aOutput.SetDimensions(output_nrows, col);
    aOutput.SetData((char *) pOutput_data, CPU);

}


template <typename T>
void linear::QRDecompositionQ(DataType &aInputA, DataType &aInputB,
                              DataType &aOutput,
                              const bool &aComplete) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    auto mem_transfer = ( operation_placement == CPU )
                        ? memory::MemoryTransfer::HOST_TO_HOST
                        : memory::MemoryTransfer::DEVICE_TO_DEVICE;

    auto row = aInputA.GetNRow();
    auto col = aInputA.GetNCol();
    auto pQr_data = (T *) aInputA.GetData(operation_placement);
    auto pQraux = (T *) aInputB.GetData(operation_placement);

    auto output_nrhs = aComplete ? row : std::min(row, col);
    auto output_size = row * output_nrhs;
    auto pOutput_data = (T *) memory::AllocateArray(output_size * sizeof(T),
                                                    operation_placement,
                                                    context);

    //TODO: revisit why added.
//    memset(pOutput_data, 0, output_size * sizeof(T));
//
//    for (auto i = 0; i < output_size; i += row + 1) {
//        pOutput_data[ i ] = 1.0f;
//    }


    memory::MemCpy((char *) pOutput_data, (char *) pQr_data,
                   ( output_size * sizeof(T)), context, mem_transfer);

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);

    auto rc = solver->Orgqr(row, output_nrhs, col, pOutput_data, row, pQraux);

    if (rc != 0) {
        delete[] pOutput_data;
        MPCR_API_EXCEPTION("Error While Performing QR.Q", rc);
    }

    aOutput.ClearUp();
    aOutput.SetSize(output_size);
    aOutput.SetDimensions(row, output_nrhs);
    aOutput.SetData((char *) pOutput_data, operation_placement);

}


template <typename T>
void
linear::ReciprocalCondition(DataType &aInput, DataType &aOutput,
                            const std::string &aNorm, const bool &aTriangle) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    auto mem_transfer = ( operation_placement == CPU )
                        ? memory::MemoryTransfer::HOST_TO_HOST
                        : memory::MemoryTransfer::DEVICE_TO_DEVICE;

    auto row = aInput.GetNRow();
    auto col = aInput.GetNCol();

    if (row != col) {
        MPCR_API_EXCEPTION("Wrong Dimensions for rcond", -1);
    }

    string norm = aNorm == "I" ? "inf" : "one";

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);

    auto pData = (T *) aInput.GetData(operation_placement);

    auto pRcond = (T *) memory::AllocateArray(1 * sizeof(T),
                                              operation_placement, context);

    if (aTriangle) {
        auto upper_triangle = false;
        auto unit_triangle = false;

        auto rc = solver->Trcon(aNorm, upper_triangle, unit_triangle, row,
                                pData, col, pRcond);
        if (rc != 0) {
            delete[] pRcond;
            MPCR_API_EXCEPTION("Error While Performing rcond Triangle", rc);
        }

    } else {

        auto pIpiv = (int64_t *) memory::AllocateArray(row * sizeof(int64_t),
                                                       operation_placement,
                                                       context);
        auto pTemp_data = (T *) memory::AllocateArray(col * row * sizeof(T),
                                                      operation_placement,
                                                      context);

        //TODO: revise to see if it's allocated on cpu or gpu incase of GPU
        T xnorm = 0;

        memory::MemCpy((char *) pTemp_data, (char *) pData,
                       ( row * col ) * sizeof(T), context, mem_transfer);

        if (norm == "one") {
            xnorm = NormMACS <T>(aInput);
        } else if (norm == "inf") {
            xnorm = NormMARS <T>(aInput);
        }

        auto rc = solver->Getrf(row, col, pTemp_data, col, pIpiv);
        if (rc != 0) {
            delete[] pRcond;
            delete[] pIpiv;
            delete[] pTemp_data;
            MPCR_API_EXCEPTION("Error While Performing rcond getrf", rc);
        }
        delete[] pIpiv;


        rc = solver->Gecon(aNorm, row, pTemp_data, col, xnorm, pRcond);

        if (rc != 0) {
            delete[] pRcond;
            delete[] pIpiv;
            delete[] pTemp_data;
            MPCR_API_EXCEPTION("Error While Performing rcond gecon", rc);
        }

        delete[] pTemp_data;
    }


    aOutput.ClearUp();
    aOutput.SetSize(1);
    aOutput.SetData((char *) pRcond, operation_placement);


}


template <typename T>
void
linear::QRDecompositionQY(DataType &aInputA, DataType &aInputB,
                          DataType &aInputC, DataType &aOutput,
                          const bool &aTranspose) {

    auto context = ContextManager::GetOperationContext();
    auto operation_placement = context->GetOperationPlacement();

    auto mem_transfer = ( operation_placement == CPU )
                        ? memory::MemoryTransfer::HOST_TO_HOST
                        : memory::MemoryTransfer::DEVICE_TO_DEVICE;

    auto row = aInputA.GetNRow();
    auto col = aInputA.GetNCol();
    auto pQr_data = (T *) aInputA.GetData(operation_placement);
    auto pQraux = (T *) aInputB.GetData(operation_placement);

    auto output_nrhs = aInputC.GetNCol();
    auto output_size = row * output_nrhs;
    auto pOutput_data = (T *) memory::AllocateArray(output_size * sizeof(T),
                                                    operation_placement,
                                                    context);

    memory::MemCpy((char *) pOutput_data, (char *) pQr_data,
                   ( output_size * sizeof(T)), context, mem_transfer);

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        operation_placement);

    auto rc = solver->Orgqr(row, output_nrhs, col, pOutput_data, row, pQraux);

    if (rc != 0) {
        delete[] pOutput_data;
        MPCR_API_EXCEPTION("Error While Performing QR.QY", rc);
    }

    aOutput.ClearUp();
    aOutput.SetSize(output_size);
    aOutput.SetDimensions(row, output_nrhs);
    aOutput.SetData((char *) pOutput_data, operation_placement);
}


SIMPLE_INSTANTIATE(void, linear::CrossProduct, DataType &aInputA,
                   DataType &aInputB, DataType &aOutput,
                   const bool &aTransposeA, const bool &aTransposeB,
                   const bool &aSymmetrize, const double &aAlpha,
                   const double &aBeta)

SIMPLE_INSTANTIATE(void, linear::IsSymmetric, DataType &aInput, bool &aOutput)

SIMPLE_INSTANTIATE(void, linear::Cholesky, DataType &aInputA,
                   DataType &aOutput, const bool &aUpperTriangle)

SIMPLE_INSTANTIATE(void, linear::CholeskyInv, DataType &aInputA,
                   DataType &aOutput, const size_t &aNCol)

SIMPLE_INSTANTIATE(void, linear::Solve, DataType &aInputA, DataType &aInputB,
                   DataType &aOutput, const bool &aSingle)

SIMPLE_INSTANTIATE(void, linear::BackSolve, DataType &aInputA,
                   DataType &aInputB, DataType &aOutput, const size_t &aCol,
                   const bool &aUpperTri, const bool &aTranspose,
                   const char &aSide, const double &aAlpha)

SIMPLE_INSTANTIATE(void, linear::Eigen, DataType &aInput,
                   DataType &aOutputValues, DataType *apOutputVectors)

SIMPLE_INSTANTIATE(void, linear::Norm, DataType &aInput,
                   const std::string &aType, DataType &aOutput)

SIMPLE_INSTANTIATE(void, linear::ReciprocalCondition, DataType &aInput,
                   DataType &aOutput, const std::string &aNorm,
                   const bool &aTriangle)

SIMPLE_INSTANTIATE(void, linear::SVD, DataType &aInputA, DataType &aOutputS,
                   DataType &aOutputU, DataType &aOutputV, const size_t &aNu,
                   const size_t &aNv, const bool &aTranspose)

SIMPLE_INSTANTIATE(void, linear::QRDecompositionQ, DataType &aInputA,
                   DataType &aInputB, DataType &aOutput, const bool &aComplete)

SIMPLE_INSTANTIATE(void, linear::QRDecomposition, DataType &aInputA,
                   DataType &aOutputQr, DataType &aOutputQraux,
                   DataType &aOutputPivot, DataType &aRank,
                   const double &aTolerance)

SIMPLE_INSTANTIATE(void, linear::QRDecompositionR, DataType &aInputA,
                   DataType &aOutput, const bool &aComplete)

SIMPLE_INSTANTIATE(void, linear::QRDecompositionQY, DataType &aInputA,
                   DataType &aInputB, DataType &aInputC, DataType &aOutput,
                   const bool &aTranspose)


