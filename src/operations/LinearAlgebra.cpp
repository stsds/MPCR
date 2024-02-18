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

#include <cusolverDn.h>


#endif

using namespace mpcr::operations;
using namespace std;


template <typename T>
void
linear::CrossProduct(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                     const bool &aTransposeA, const bool &aTransposeB,
                     const bool &aSymmetrize, const double &aAlpha,
                     const double &aBeta) {

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

    auto pData_a = (T *) aInputA.GetData();
    auto pData_b = (T *) aInputB.GetData();

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
        pData_out = (T *) aOutput.GetData();

        if (aOutput.GetNRow() != row_a || aOutput.GetNCol() != col_b) {
            MPCR_API_EXCEPTION("Wrong Output Matrix Dimensions", -1);
        }

    } else {

        auto output_size = row_a * col_b;
        pData_out = new T[output_size];
        memset(pData_out, 0, sizeof(T) * output_size);
        aOutput.ClearUp();
        aOutput.SetSize(output_size);
        aOutput.SetDimensions(row_a, col_b);
    }

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        CPU);

    if (!is_one_input) {
        solver->Gemm(aTransposeA, aTransposeB, row_a, col_b, col_a, aAlpha,
                     pData_a, lda, pData_b, ldb, aBeta, pData_out, row_a);
    } else {
        solver->Syrk(true, aTransposeA, row_a, col_a, aAlpha, pData_a, lda,
                     aBeta, pData_out, row_a);

    }

    aOutput.SetData((char *) pData_out);

    if (is_one_input && aSymmetrize) {
        // this kernel will need to be implemented in GPU too.
        Symmetrize <T>(aOutput, true);
    }

    if (flag_conv) {
        aInputB.ToVector();
    }

}


template <typename T>
void
linear::IsSymmetric(DataType &aInput, bool &aOutput) {

    aOutput = false;
    auto pData = (T *) aInput.GetData();
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

    auto row = aInputA.GetNRow();
    auto col = aInputA.GetNCol();


    if (row != col) {
        MPCR_API_EXCEPTION(
            "Cannot Apply Cholesky Decomposition on non-square Matrix", -1);
    }

    auto pOutput = new T[row * col];
    auto pData = (T *) aInputA.GetData();
    memcpy(pOutput, pData, ( row * col * sizeof(T)));

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        CPU);
    auto rc = solver->Potrf(aUpperTriangle, row, pOutput, row);

    if (rc != 0) {
        MPCR_API_EXCEPTION(
            "Error While Applying Cholesky Decomposition", rc);
    }


    aOutput.ClearUp();
    aOutput.SetDimensions(aInputA);
    aOutput.SetData((char *) pOutput);
    aOutput.FillTriangle(0, !aUpperTriangle);

}


template <typename T>
void
linear::CholeskyInv(DataType &aInputA, DataType &aOutput, const size_t &aNCol) {

    auto pData = (T *) aInputA.GetData();
    auto col = aInputA.GetNCol();

    if (aNCol > col) {
        MPCR_API_EXCEPTION(
            "Size Cannot exceed the Number of Cols of Input", -1);
    }

    T *pOutput = nullptr;
    aOutput.ClearUp();
    if (aNCol == col) {
        aOutput = aInputA;
        aOutput.SetDimensions(aNCol, aNCol);
        pOutput = (T *) aOutput.GetData();
    } else {
        auto new_size = aNCol * aNCol;
        aOutput.SetSize(new_size);
        aOutput.SetDimensions(aNCol, aNCol);
        auto pTemp_data = new T[new_size];
        size_t idx;
        for (auto i = 0; i < aNCol; i++) {
            for (auto j = 0; j < aNCol; j++) {
                idx = j + ( i * aNCol );
                pTemp_data[ idx ] = pData[ j + ( col * i ) ];
            }
        }
        pOutput = pTemp_data;
    }


    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        CPU);
    auto rc = solver->Potri(true, aNCol, pOutput, aOutput.GetNRow());

    if (rc != 0) {
        MPCR_API_EXCEPTION(
            "Error While Applying Cholesky Decomposition", rc);
    }


    aOutput.SetData((char *) pOutput);
    Symmetrize <T>(aOutput, false);

}


template <typename T>
void linear::Solve(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                   const bool &aSingle) {

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

    auto pIpiv = new int64_t[cols_a];
    aOutput.ClearUp();
    auto rc = 0;


    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        CPU);

    if (!aSingle) {
        DataType dump = aInputA;
        aOutput = aInputB;
        auto pData_dump = (T *) dump.GetData();
        auto pData_in_out = (T *) aOutput.GetData();

        rc = solver->Gesv(cols_a, cols_b, pData_dump, rows_a, (void *) pIpiv,
                          pData_in_out, rows_b);

    } else {
        aOutput = aInputA;
        auto pData_in_out = (T *) aOutput.GetData();

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

    auto pData = (T *) aInputA.GetData();
    auto pData_b = (T *) aInputB.GetData();
    auto pData_in_out = new T[col_b * aCol];

    for (auto i = 0; i < col_b; i++) {
        memcpy(( pData_in_out + ( aCol * i )), pData_b + ( row_b * i ),
               ( sizeof(T) * aCol ));
    }

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        CPU);

    solver->Trsm(left_side, aUpperTri, aTranspose, row_b, col_b, aAlpha, pData,
                 row_a, pData_in_out, row_b);


    aOutput.SetData((char *) pData_in_out);
    if (flag_transform) {
        aInputB.ToVector();
    }


}


template <typename T>
void
linear::SVD(DataType &aInputA, DataType &aOutputS, DataType &aOutputU,
            DataType &aOutputV, const size_t &aNu,
            const size_t &aNv, const bool &aTranspose) {


    //s ,u ,vt
    auto row = aInputA.GetNRow();
    auto col = aInputA.GetNCol();
    auto pData = (T *) aInputA.GetData();

    auto min_dim = std::min(row, col);
    auto pOutput_s = new T[min_dim];
    T *pOutput_u = nullptr;
    T *pOutput_vt = nullptr;

    aOutputS.ClearUp();
    aOutputU.ClearUp();
    aOutputV.ClearUp();

    aOutputS.SetSize(min_dim);

    if (aNu) {
        pOutput_u = new T[row * aNu];
        aOutputU.SetSize(row * aNu);
        aOutputU.SetDimensions(row, aNu);
    }

    if (aNv) {
        pOutput_vt = new T[col * aNv];
        aOutputV.SetSize(col * aNv);
        /** Will be transposed at the end in case of svd **/
        aOutputV.SetDimensions(aNv, col);
    }


    auto pTemp_data = new T[row * col];
    memcpy((void *) pTemp_data, (void *) pData, ( row * col ) * sizeof(T));

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
        CPU);

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


    aOutputS.SetData((char *) pOutput_s);
    aOutputV.SetData((char *) pOutput_vt);
    aOutputU.SetData((char *) pOutput_u);
    if (aTranspose) {
        aOutputV.Transpose();
    }

}


template <typename T>
void linear::Eigen(DataType &aInput, DataType &aOutputValues,
                   DataType *apOutputVectors) {

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

    auto pData = (T *) aInput.GetData();

    auto pValues = new T[col];
    auto pVectors = new T[col * col];

    memcpy((char *) pVectors, (char *) pData, col * col * sizeof(T));

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        CPU);

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
        apOutputVectors->SetData((char *) pVectors);
        ReverseMatrix <T>(*apOutputVectors);
    } else {
        delete[] pVectors;
    }

    std::reverse(pValues, pValues + col);
    aOutputValues.ClearUp();
    aOutputValues.SetSize(col);
    aOutputValues.SetData((char *) pValues);


}


template <typename T>
void
linear::Norm(DataType &aInput, const std::string &aType, DataType &aOutput) {

    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();
    aOutput.ClearUp();
    aOutput.SetSize(1);

    auto pOutput = new T[1];

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

    aOutput.SetData((char *) pOutput);
}


template <typename T>
void
linear::QRDecomposition(DataType &aInputA, DataType &aOutputQr,
                        DataType &aOutputQraux, DataType &aOutputPivot,
                        DataType &aRank, const double &aTolerance) {

    auto col = aInputA.GetNCol();
    auto row = aInputA.GetNRow();
    auto min_dim = std::min(col, row);
    auto pData = (T *) aInputA.GetData();

    auto pQr_in_out = new T[row * col];
    auto pQraux = new T[min_dim];
    auto pJpvt = new int64_t[col];

    memset(pJpvt, 0, col * sizeof(int64_t));

    memcpy((void *) pQr_in_out, (void *) pData,
           ( aInputA.GetSize()) * sizeof(T));

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        CPU);

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
    aOutputQr.SetData((char *) pQr_in_out);

    aOutputQraux.SetSize(min_dim);
    aOutputQraux.SetData((char *) pQraux);

    auto pTemp_pvt = new T[col];


    std::copy(pJpvt, pJpvt + col, pTemp_pvt);
    delete[] pJpvt;

    aOutputPivot.SetSize(col);
    aOutputPivot.SetData((char *) pTemp_pvt);

    auto pRank = new T[1];
    GetRank <T>(aOutputQr, aTolerance, *pRank);

    aRank.ClearUp();
    aRank.SetSize(1);
    aRank.SetData((char *) pRank);

}


template <typename T>
void
linear::QRDecompositionR(DataType &aInputA, DataType &aOutput,
                         const bool &aComplete) {

    auto col = aInputA.GetNCol();
    auto row = aInputA.GetNRow();
    auto output_nrows = aComplete ? row : std::min(row, col);
    auto output_size = output_nrows * col;
    auto pOutput_data = new T[output_size];
    auto pData = (T *) aInputA.GetData();

    memset(pOutput_data, 0, output_size * sizeof(T));

    for (auto j = 0; j < col; j++) {
        for (auto i = 0; i <= j && i < output_nrows; i++)
            pOutput_data[ i + output_nrows * j ] = pData[ i + row * j ];
    }

    aOutput.ClearUp();
    aOutput.SetSize(output_size);
    aOutput.SetDimensions(output_nrows, col);
    aOutput.SetData((char *) pOutput_data);

}


template <typename T>
void linear::QRDecompositionQ(DataType &aInputA, DataType &aInputB,
                              DataType &aOutput,
                              const bool &aComplete) {

    auto row = aInputA.GetNRow();
    auto col = aInputA.GetNCol();
    auto pQr_data = (T *) aInputA.GetData();
    auto pQraux = (T *) aInputB.GetData();

    auto output_nrhs = aComplete ? row : std::min(row, col);
    auto output_size = row * output_nrhs;
    auto pOutput_data = new T[output_size];

    memset(pOutput_data, 0, output_size * sizeof(T));

    for (auto i = 0; i < output_size; i += row + 1) {
        pOutput_data[ i ] = 1.0f;
    }


    memcpy((void *) pOutput_data, (void *) pQr_data,
           ( output_size * sizeof(T)));

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        CPU);

    auto rc = solver->Orgqr(row, output_nrhs, col, pOutput_data, row, pQraux);

    if (rc != 0) {
        delete[] pOutput_data;
        MPCR_API_EXCEPTION("Error While Performing QR.Q", rc);
    }

    aOutput.ClearUp();
    aOutput.SetSize(output_size);
    aOutput.SetDimensions(row, output_nrhs);
    aOutput.SetData((char *) pOutput_data);

}


template <typename T>
void
linear::ReciprocalCondition(DataType &aInput, DataType &aOutput,
                            const std::string &aNorm, const bool &aTriangle) {

    auto row = aInput.GetNRow();
    auto col = aInput.GetNCol();
    auto pData = (T *) aInput.GetData();
    string norm = aNorm == "I" ? "inf" : "one";

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        CPU);


    if (row != col) {
        MPCR_API_EXCEPTION("Wrong Dimensions for rcond", -1);
    }
    auto pRcond = new T[1];

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

        auto pIpiv = new int64_t[row];
        auto pTemp_data = new T[row * col];
        T xnorm = 0;
        memcpy((void *) pTemp_data, (void *) pData, ( row * col ) * sizeof(T));

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
    aOutput.SetData((char *) pRcond);


}


template <typename T>
void
linear::QRDecompositionQY(DataType &aInputA, DataType &aInputB,
                          DataType &aInputC, DataType &aOutput,
                          const bool &aTranspose) {

    auto row = aInputA.GetNRow();
    auto col = aInputA.GetNCol();
    auto pQr_data = (T *) aInputA.GetData();
    auto pQraux = (T *) aInputB.GetData();

    auto output_nrhs = aInputC.GetNCol();
    auto output_size = row * output_nrhs;
    auto pOutput_data = new T[output_size];

    memcpy((void *) pOutput_data, (void *) pQr_data,
           ( output_size * sizeof(T)));

    auto solver = linear::LinearAlgebraBackendFactory <T>::CreateBackend(
        CPU);

    auto rc = solver->Orgqr(row, output_nrhs, col, pOutput_data, row, pQraux);

    if (rc != 0) {
        delete[] pOutput_data;
        MPCR_API_EXCEPTION("Error While Performing QR.QY", rc);
    }

    aOutput.ClearUp();
    aOutput.SetSize(output_size);
    aOutput.SetDimensions(row, output_nrhs);
    aOutput.SetData((char *) pOutput_data);
}
//
//#ifdef USE_CUDA
//template <typename T>
//void
//linear::CudaCholesky(DataType &aInputA, DataType &aOutput,
//                     const bool &aUpperTriangle) {
//
//    auto row = aInputA.GetNRow();
//    auto col = aInputA.GetNCol();
//    auto triangle = aUpperTriangle ? cublasFillMode_t::CUBLAS_FILL_MODE_UPPER : cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
//
//
//    if (row != col) {
//        MPCR_API_EXCEPTION(
//            "Cannot Apply Cholesky Decomposition on non-square Matrix", -1);
//    }
//
//    aInputA.Print();
//    T* pOutput = nullptr;
//    T* pOutput_out = new T[aInputA.GetSize()];
//
//
//    GPU_ERROR_CHECK(cudaMalloc((void**) &pOutput, row * col * sizeof(T)));
//    auto pData = (T *) aInputA.GetData();
//    GPU_ERROR_CHECK(cudaMemcpy(pOutput, pData, row * col * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
//
//
//    cusolverDnHandle_t cuHandle;
//    cusolverDnCreate(&cuHandle);
//    int* devInfo;
//    int lWork = 0;
//    cudaMalloc((void**)&devInfo, sizeof(int));
//    T* workspace = nullptr;
//
//
//    if constexpr(is_double<T>()) {
////
////        cusolverDnDpotrf_bufferSize(cuHandle,
////                                    triangle,
////                                    row,
////                                    pOutput,
////                                    row,
////                                    &lWork);
////
////
////        GPU_ERROR_CHECK(cudaMalloc(&workspace, lWork * sizeof(T)));
//
//        cusolverDnDpotrf(cuHandle,
//                         triangle,
//                         row,
//                         pOutput,
//                         row,
//                         workspace,
//                         lWork,
//                         devInfo);
//    }
//    else {
//        cusolverDnSpotrf(cuHandle,
//                         triangle,
//                         row,
//                         pOutput,
//                         row,
//                         nullptr,
//                         0,
//                         devInfo);
//    }
//
//
//    // Check for errors
//    int devInfo_h = 0;
//    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
//    if (devInfo_h != 0) {
//        exit(1);
//    }
//
//
//
//    GPU_ERROR_CHECK(cudaMemcpy(pOutput_out, pOutput, row * col * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
//
//    cudaFree(devInfo);
//    cudaFree(workspace);
//    cudaFree(pOutput);
//    cusolverDnDestroy(cuHandle);
//
//    aOutput.ClearUp();
//    aOutput.SetDimensions(aInputA);
//    aOutput.SetData((char *) pOutput_out);
//
////    aOutput.FillTriangle(0, !aUpperTriangle);
//
//}
//
//
//FLOATING_POINT_INST(void, linear::CudaCholesky, DataType &aInputA,
//                    DataType &aOutput, const bool &aUpperTriangle)
//
//#endif

FLOATING_POINT_INST(void, linear::CrossProduct, DataType &aInputA,
                    DataType &aInputB, DataType &aOutput,
                    const bool &aTransposeA, const bool &aTransposeB,
                    const bool &aSymmetrize, const double &aAlpha,
                    const double &aBeta)

FLOATING_POINT_INST(void, linear::IsSymmetric, DataType &aInput, bool &aOutput)

FLOATING_POINT_INST(void, linear::Cholesky, DataType &aInputA,
                    DataType &aOutput, const bool &aUpperTriangle)

FLOATING_POINT_INST(void, linear::CholeskyInv, DataType &aInputA,
                    DataType &aOutput, const size_t &aNCol)

FLOATING_POINT_INST(void, linear::Solve, DataType &aInputA, DataType &aInputB,
                    DataType &aOutput, const bool &aSingle)

FLOATING_POINT_INST(void, linear::BackSolve, DataType &aInputA,
                    DataType &aInputB, DataType &aOutput, const size_t &aCol,
                    const bool &aUpperTri, const bool &aTranspose,
                    const char &aSide, const double &aAlpha)

FLOATING_POINT_INST(void, linear::Eigen, DataType &aInput,
                    DataType &aOutputValues, DataType *apOutputVectors)

FLOATING_POINT_INST(void, linear::Norm, DataType &aInput,
                    const std::string &aType, DataType &aOutput)

FLOATING_POINT_INST(void, linear::ReciprocalCondition, DataType &aInput,
                    DataType &aOutput, const std::string &aNorm,
                    const bool &aTriangle)

FLOATING_POINT_INST(void, linear::SVD, DataType &aInputA, DataType &aOutputS,
                    DataType &aOutputU, DataType &aOutputV, const size_t &aNu,
                    const size_t &aNv, const bool &aTranspose)

FLOATING_POINT_INST(void, linear::QRDecompositionQ, DataType &aInputA,
                    DataType &aInputB, DataType &aOutput, const bool &aComplete)

FLOATING_POINT_INST(void, linear::QRDecomposition, DataType &aInputA,
                    DataType &aOutputQr, DataType &aOutputQraux,
                    DataType &aOutputPivot, DataType &aRank,
                    const double &aTolerance)

FLOATING_POINT_INST(void, linear::QRDecompositionR, DataType &aInputA,
                    DataType &aOutput, const bool &aComplete)

FLOATING_POINT_INST(void, linear::QRDecompositionQY, DataType &aInputA,
                    DataType &aInputB, DataType &aInputC, DataType &aOutput,
                    const bool &aTranspose)


