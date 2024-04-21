

#include <operations/concrete/GPULinearAlgerba.hpp>
#include <kernels/ContextManager.hpp>
#include <cusolverDn.h>
#include <utilities/TypeChecker.hpp>
#include <kernels/MemoryHandler.hpp>


using namespace mpcr::operations::linear;
using namespace mpcr::kernels;


template <typename T>
void
GPULinearAlgebra <T>::Gemm(const bool &aTransposeA,
                           const bool &aTransposeB,
                           const int &aNumRowsA,
                           const int &aNumColB,
                           const int &aNumRowB,
                           const T &aAlpha, const T *apDataA,
                           const int &aLda, const T *apDataB,
                           const int &aLdb, const T &aBeta,
                           T *apDataC, const int &aLdc) {

    auto context = ContextManager::GetOperationContext();
    auto transpose_a = aTransposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto transpose_b = aTransposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto cublas_handle = context->GetCuBlasDnHandle();

    auto rc = 0;

    if constexpr(is_double <T>()) {
        rc = cublasDgemm(cublas_handle, transpose_a, transpose_b,
                         aNumRowsA,
                         aNumColB, aNumRowB, &aAlpha, apDataA, aLda,
                         apDataB, aLdb, &aBeta, apDataC, aLdc);
    } else {
        rc = cublasSgemm(cublas_handle, transpose_a, transpose_b,
                         aNumRowsA,
                         aNumColB, aNumRowB, &aAlpha, apDataA, aLda,
                         apDataB, aLdb, &aBeta, apDataC, aLdc);
    }

    if (rc != 0) {
        MPCR_API_EXCEPTION("Error While Performing Gemm on GPU", rc);
    }

}


template <typename T>
void
GPULinearAlgebra <T>::Syrk(const bool &aFillLower,
                           const bool &aTranspose,
                           const int &aNumRowA,
                           const int &aNumColA,
                           const T &aAlpha, const T *apDataA,
                           const int &aLda, const T &aBeta,
                           T *apDataC, const int &aLdc) {

    auto context = ContextManager::GetOperationContext();
    auto cublas_handle = context->GetCuBlasDnHandle();
    auto triangle = aFillLower ? CUBLAS_FILL_MODE_LOWER
                               : CUBLAS_FILL_MODE_UPPER;
    auto transpose = aTranspose ? CUBLAS_OP_T : CUBLAS_OP_N;

    int rc = 0;

    if constexpr(is_double <T>()) {
        rc = cublasDsyrk(cublas_handle, triangle, transpose, aNumRowA,
                         aNumColA, &aAlpha, apDataA, aLda, &aBeta, apDataC,
                         aLdc);

    } else {
        rc = cublasSsyrk(cublas_handle, triangle, transpose, aNumRowA,
                         aNumColA, &aAlpha, apDataA, aLda, &aBeta, apDataC,
                         aLdc);
    }

    if (rc != 0) {
        MPCR_API_EXCEPTION("Error While Performing Syrk on GPU", rc);
    }


}


template <typename T>
void
GPULinearAlgebra <T>::Trsm(const bool &aLeftSide,
                           const bool &aFillUpperTri,
                           const bool &aTranspose,
                           const int &aNumRowsB,
                           const int &aNumColsB,
                           const T &aAlpha, const T *apDataA,
                           const int &aLda, T *apDataB,
                           const int &aLdb) {

    auto context = ContextManager::GetOperationContext();
    auto cublas_handle = context->GetCuBlasDnHandle();
    auto side = aLeftSide ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    auto transpose = aTranspose ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto triangle = aFillUpperTri ? CUBLAS_FILL_MODE_UPPER
                                  : CUBLAS_FILL_MODE_LOWER;
    auto diag = CUBLAS_DIAG_NON_UNIT;

    auto rc = 0;

    if constexpr(is_double <T>()) {
        rc = cublasDtrsm(cublas_handle, side, triangle, transpose, diag,
                         aNumRowsB, aNumColsB, &aAlpha, apDataA, aLda,
                         apDataB, aLdb);
    } else {
        rc = cublasStrsm(cublas_handle, side, triangle, transpose, diag,
                         aNumRowsB, aNumColsB, &aAlpha, apDataA, aLda,
                         apDataB, aLdb);
    }

    if (rc != 0) {
        MPCR_API_EXCEPTION("Error While Performing Trsm on GPU", rc);
    }
}


template <typename T>
int
GPULinearAlgebra <T>::Potrf(const bool &aFillUpperTri,
                            const int &aNumRow, T *apDataA,
                            const int &aLda) {
    auto triangle = aFillUpperTri ? CUBLAS_FILL_MODE_UPPER
                                  : CUBLAS_FILL_MODE_LOWER;
    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();
    size_t lWork_device = 0;
    size_t lWork_host = 0;
    auto data_type = is_double <T>() ? CUDA_R_64F : CUDA_R_32F;

    cusolverDnXpotrf_bufferSize(cusolver_handle, NULL, triangle, aNumRow,
                                data_type, (void *) apDataA, aNumRow, data_type,
                                &lWork_device, &lWork_host);

    auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);
    auto work_space_host = context->RequestWorkBufferHost(lWork_host);


    cusolverDnXpotrf(cusolver_handle, NULL, triangle, aNumRow, data_type,
                     (void *) apDataA, aNumRow, data_type, work_space_dev,
                     lWork_device, work_space_host, lWork_host,
                     context->GetInfoPointer());

    int rc = 0;
    memory::MemCpy((char *) &rc, (char *) context->GetInfoPointer(),
                   sizeof(int), context,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    if (context->GetRunMode() == kernels::RunMode::SYNC) {
        context->FreeWorkBufferHost();
    }

    return rc;


}


template <typename T>
int
GPULinearAlgebra <T>::Potri(const bool &aFillUpperTri,
                            const int &aNumRow, T *apDataA,
                            const int &aLda) {
    auto triangle = aFillUpperTri ? CUBLAS_FILL_MODE_UPPER
                                  : CUBLAS_FILL_MODE_LOWER;
    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();
    int lWork = 0;

    if constexpr(is_double <T>()) {
        cusolverDnDpotri_bufferSize(cusolver_handle,
                                    triangle,
                                    aNumRow,
                                    apDataA,
                                    aNumRow,
                                    &lWork);
        auto work_space = context->RequestWorkBufferDevice(lWork * sizeof(T));

        cusolverDnDpotri(cusolver_handle,
                         triangle,
                         aNumRow,
                         apDataA,
                         aNumRow,
                         (T *) work_space,
                         lWork,
                         context->GetInfoPointer());
    } else {
        cusolverDnSpotri_bufferSize(cusolver_handle,
                                    triangle,
                                    aNumRow,
                                    apDataA,
                                    aNumRow,
                                    &lWork);

        auto work_space = context->RequestWorkBufferDevice(lWork * sizeof(T));

        cusolverDnSpotri(cusolver_handle,
                         triangle,
                         aNumRow,
                         apDataA,
                         aNumRow,
                         (T *) work_space,
                         0,
                         context->GetInfoPointer());
    }

    int rc = 0;
    memory::MemCpy((char *) &rc, (char *) context->GetInfoPointer(),
                   sizeof(int), context,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    return rc;
}


template <typename T>
int
GPULinearAlgebra <T>::Gesv(const int &aNumN, const int &aNumNRH,
                           T *apDataA, const int &aLda, void *apIpiv,
                           T *apDataB,
                           const int &aLdb, T *apDataOut, const int &aLdo) {
    // TODO : This function can support half precision internally for the LU factorization

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();
    size_t lWork_device = 0;
    cusolver_int_t nitr = 0;

    if constexpr(is_double <T>()) {

        cusolverDnDDgesv_bufferSize(cusolver_handle, aNumN, aNumNRH, apDataA,
                                    aLda, (cusolver_int_t *) apIpiv, apDataB,
                                    aLdb, apDataOut, aLdo, nullptr,
                                    &lWork_device);

        auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);


        cusolverDnDDgesv(cusolver_handle, aNumN, aNumNRH, apDataA, aLda,
                         (cusolver_int_t *) apIpiv, apDataB, aLdb, apDataOut,
                         aLdo,
                         work_space_dev, lWork_device, &nitr,
                         context->GetInfoPointer());


    } else {
        cusolverDnSSgesv_bufferSize(cusolver_handle, aNumN, aNumNRH, apDataA,
                                    aLda, (cusolver_int_t *) apIpiv, apDataB,
                                    aLdb, apDataOut, aLdo, nullptr,
                                    &lWork_device);

        auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);


        cusolverDnSSgesv(cusolver_handle, aNumN, aNumNRH, apDataA, aLda,
                         (cusolver_int_t *) apIpiv, apDataB, aLdb, apDataOut,
                         aLdo,
                         work_space_dev, lWork_device, &nitr,
                         context->GetInfoPointer());
    }

    if (nitr < 0) {
        MPCR_API_EXCEPTION("Error While Performing factorization in Gesv GPU",
                           nitr);
    }

    int rc = 0;
    memory::MemCpy((char *) &rc, (char *) context->GetInfoPointer(),
                   sizeof(int), context,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    return rc;

}


template <typename T>
int
GPULinearAlgebra <T>::Getrf(const int &aNumRow, const int &aNumCol,
                            T *apDataA, const int &aLda,
                            int64_t *apIpiv) {

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();

    size_t lWork_device = 0;
    size_t lWork_host = 0;
    auto data_type = is_double <T>() ? CUDA_R_64F : CUDA_R_32F;

    cusolverDnXgetrf_bufferSize(cusolver_handle, NULL, aNumRow, aNumCol,
                                data_type, (void *) apDataA, aLda, data_type,
                                &lWork_device, &lWork_host);

    auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);
    auto work_space_host = context->RequestWorkBufferHost(lWork_host);

    cusolverDnXgetrf(cusolver_handle, NULL, aNumRow, aNumCol, data_type,
                     apDataA,
                     aLda, apIpiv, data_type, work_space_dev, lWork_device,
                     work_space_host, lWork_host, context->GetInfoPointer());


    int rc = 0;
    memory::MemCpy((char *) &rc, (char *) context->GetInfoPointer(),
                   sizeof(int), context,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    if (context->GetRunMode() == kernels::RunMode::SYNC) {
        context->FreeWorkBufferHost();
    }

    return rc;
}


template <typename T>
int
GPULinearAlgebra <T>::Getri(const int &aMatRank, T *apDataA, const int &aLda,
                            int64_t *apIpiv) {
    MPCR_API_EXCEPTION("No Getri implementation for GPU", -1);
}


template <typename T>
int
GPULinearAlgebra <T>::SVD(const signed char &aJob,
                          const int &aNumRow,
                          const int &aNumCol, T *apDataA,
                          const int &aLda, T *apDataS,
                          T *apDataU, const int &aLdu,
                          T *apDataVT, const int &aLdvt) {

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();

    size_t lWork_device = 0;
    size_t lWork_host = 0;
    auto data_type = is_double <T>() ? CUDA_R_64F : CUDA_R_32F;

    cusolverDnXgesvd_bufferSize(cusolver_handle, NULL, aJob, aJob, aNumRow,
                                aNumCol, data_type, (void *) apDataA, aLda,
                                data_type, (void *) apDataS, data_type,
                                (void *) apDataU, aLdu, data_type,
                                (void *) apDataVT, aLdvt, data_type,
                                &lWork_device, &lWork_host);

    auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);
    auto work_space_host = context->RequestWorkBufferHost(lWork_host);

    cusolverDnXgesvd(cusolver_handle, NULL, aJob, aJob, aNumRow, aNumCol,
                     data_type, (void *) apDataA, aLda, data_type,
                     (void *) apDataS, data_type, (void *) apDataU, aLdu,
                     data_type, (void *) apDataVT, aLdvt, data_type,
                     work_space_dev, lWork_device, work_space_host, lWork_host,
                     context->GetInfoPointer());

    int rc = 0;
    memory::MemCpy((char *) &rc, (char *) context->GetInfoPointer(),
                   sizeof(int), context,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    if (context->GetRunMode() == kernels::RunMode::SYNC) {
        context->FreeWorkBufferHost();
    }

    return rc;
}


template <typename T>
int
GPULinearAlgebra <T>::Syevd(const bool &aJobzNoVec,
                            const bool &aFillUpperTri,
                            const int &aNumCol, T *apDataA,
                            const int64_t &aLda, T *apDataW) {

    auto jobz = aJobzNoVec ? CUSOLVER_EIG_MODE_NOVECTOR
                           : CUSOLVER_EIG_MODE_VECTOR;

    auto triangle = aFillUpperTri ? CUBLAS_FILL_MODE_UPPER
                                  : CUBLAS_FILL_MODE_LOWER;

    auto data_type = is_double <T>() ? CUDA_R_64F : CUDA_R_32F;

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();

    size_t lWork_device = 0;
    size_t lWork_host = 0;

    cusolverDnXsyevd_bufferSize(cusolver_handle, NULL, jobz, triangle, aNumCol,
                                data_type, apDataA, aLda, data_type, apDataW,
                                data_type, &lWork_device, &lWork_host);

    auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);
    auto work_space_host = context->RequestWorkBufferHost(lWork_host);

    cusolverDnXsyevd(cusolver_handle, NULL, jobz, triangle, aNumCol, data_type,
                     apDataA, aLda, data_type, apDataW, data_type,
                     work_space_dev,
                     lWork_device, work_space_host, lWork_host,
                     context->GetInfoPointer());

    int rc = 0;
    memory::MemCpy((char *) &rc, (char *) context->GetInfoPointer(),
                   sizeof(int), context,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    if (context->GetRunMode() == kernels::RunMode::SYNC) {
        context->FreeWorkBufferHost();
    }

    return rc;

}


template <typename T>
int
GPULinearAlgebra <T>::Geqp3(const int &aNumRow, const int &aNumCol, T *apDataA,
                            const int &aLda, int64_t *aJpVt, T *aTaw) {

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();

    size_t lWork_device = 0;
    size_t lWork_host = 0;

    auto data_type = is_double <T>() ? CUDA_R_64F : CUDA_R_32F;

    cusolverDnXgeqrf_bufferSize(cusolver_handle, NULL, aNumRow, aNumCol,
                                data_type, apDataA, aLda, data_type, aTaw,
                                data_type, &lWork_device, &lWork_host);

    auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);
    auto work_space_host = context->RequestWorkBufferHost(lWork_host);

    cusolverDnXgeqrf(cusolver_handle, NULL, aNumRow, aNumCol, data_type,
                     apDataA,
                     aLda, data_type, aTaw, data_type, work_space_dev,
                     lWork_device, work_space_host, lWork_host,
                     context->GetInfoPointer());

    int rc = 0;
    memory::MemCpy((char *) &rc, (char *) context->GetInfoPointer(),
                   sizeof(int), context,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    if (context->GetRunMode() == kernels::RunMode::SYNC) {
        context->FreeWorkBufferHost();
    }

    return rc;

}


template <typename T>
int
GPULinearAlgebra <T>::Orgqr(const int &aNumRow, const int &aNum,
                            const int &aNumCol, T *apDataA,
                            const int &aLda, const T *aTau) {

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();
    int lWork_device = 0;

    if constexpr(is_double <T>()) {
        cusolverDnDorgqr_bufferSize(cusolver_handle, aNumRow, aNum, aNumCol,
                                    apDataA, aLda, aTau, &lWork_device);

        auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);

        cusolverDnDorgqr(cusolver_handle, aNumRow, aNum, aNumCol, apDataA, aLda,
                         aTau, (double *) work_space_dev, lWork_device,
                         context->GetInfoPointer());
    } else {
        cusolverDnSorgqr_bufferSize(cusolver_handle, aNumRow, aNum, aNumCol,
                                    apDataA, aLda, aTau, &lWork_device);

        auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);

        cusolverDnSorgqr(cusolver_handle, aNumRow, aNum, aNumCol, apDataA, aLda,
                         aTau, (float *) work_space_dev, lWork_device,
                         context->GetInfoPointer());
    }

    int rc = 0;
    memory::MemCpy((char *) &rc, (char *) context->GetInfoPointer(),
                   sizeof(int), context,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    return rc;
}


template <typename T>
int
GPULinearAlgebra <T>::Gecon(const std::string &aNorm, const int &aNumRow,
                            const T *apData, const int &aLda, T aNormVal,
                            T *aRCond) {

    auto context = ContextManager::GetOperationContext();
    auto buf_size = aNumRow * aNumRow * sizeof(T);
    auto pData_copy = memory::AllocateArray(buf_size, GPU, context);

    memory::MemCpy(pData_copy, (char *) apData, buf_size, context,
                   memory::MemoryTransfer::DEVICE_TO_DEVICE);

    auto rc = this->SVD('A', aNumRow, aNumRow, (T *) pData_copy, aLda, nullptr,
                        nullptr, 1, nullptr, 1);

    if (rc != 0) {
        MPCR_API_EXCEPTION("Error while Performing Gecon, SVD ", rc);
    }

    auto pData_host = memory::AllocateArray(buf_size, CPU, context);
    memory::MemCpy(pData_host, pData_copy, buf_size, context,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    auto data = (T *) pData_host;

    *aRCond = data[ 0 ] / data[ aNumRow - 1 ];

    memory::DestroyArray(pData_host, CPU, context);
    memory::DestroyArray(pData_copy, GPU, context);

}


template <typename T>
int GPULinearAlgebra <T>::Trcon(const std::string &aNorm,
                                const bool &aUpperTriangle,
                                const bool &aUnitTriangle, const int &aMatOrder,
                                const T *apData, const int &aLda, T *aRCond) {
    // NO GPU Implementation
    MPCR_API_EXCEPTION("No Trcon implementation for GPU", -1);
}


template <typename T>
int GPULinearAlgebra <T>::Getrs(const bool &aTransposeA, const size_t &aNumRowA,
                                const size_t &aNumRhs, const T *apDataA,
                                const size_t &aLda, const int64_t *apIpiv,
                                T *apDataB, const size_t &aLdb) {

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();
    auto transpose = aTransposeA ? CUBLAS_OP_T : CUBLAS_OP_N;

    auto data_type = is_double <T>() ? CUDA_R_64F : CUDA_R_32F;


    cusolverDnXgetrs(cusolver_handle, NULL, transpose, aNumRowA, aNumRhs,
                     data_type, (void *) apDataA, aLda, apIpiv, data_type,
                     (void *) apDataB, aLdb, context->GetInfoPointer());


    int rc = 0;
    memory::MemCpy((char *) &rc, (char *) context->GetInfoPointer(),
                   sizeof(int), context,
                   memory::MemoryTransfer::DEVICE_TO_HOST);

    if (context->GetRunMode() == kernels::RunMode::SYNC) {
        context->FreeWorkBufferHost();
    }

    return rc;

}


#ifdef USING_HALF


template <typename T>
void
GPULinearAlgebra <T>::HalfGemm(const bool &aTransposeA, const bool &aTransposeB,
                               const int &aNumRowsA, const int &aNumColB,
                               const int &aNumRowB, const half &aAlpha,
                               const half *apDataA, const int &aLda,
                               const half *apDataB, const int &aLdb,
                               const half &aBeta, half *apDataC,
                               const int &aLdc) {

    auto context = ContextManager::GetOperationContext();
    auto transpose_a = aTransposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto transpose_b = aTransposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto cublas_handle = context->GetCuBlasDnHandle();

    auto rc = 0;

    rc = cublasHgemm(cublas_handle, transpose_a, transpose_b, aNumRowsA,
                     aNumColB, aNumRowB, &aAlpha, apDataA, aLda, apDataB, aLdb,
                     &aBeta, apDataC, aLdc);


    if (rc != 0) {
        MPCR_API_EXCEPTION("Error While Performing Gemm on GPU", rc);
    }

}


#endif
