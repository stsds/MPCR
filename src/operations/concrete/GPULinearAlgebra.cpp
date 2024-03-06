

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
                           const T &aAlpha, const T *aDataA,
                           const int &aLda, const T *aDataB,
                           const int &aLdb, const T &aBeta,
                           T *aDataC, const int &aLdc) {

    auto context = ContextManager::GetOperationContext();
    auto transpose_a = aTransposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto transpose_b = aTransposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto cublas_handle = context->GetCuBlasDnHandle();

    auto rc = 0;

    if constexpr(is_double <T>()) {
        rc = cublasDgemm(cublas_handle, transpose_a, transpose_b,
                         aNumRowsA,
                         aNumColB, aNumRowB, &aAlpha, aDataA, aLda,
                         aDataB, aLdb, &aBeta, aDataC, aLdc);
    } else {
        rc = cublasSgemm(cublas_handle, transpose_a, transpose_b,
                         aNumRowsA,
                         aNumColB, aNumRowB, &aAlpha, aDataA, aLda,
                         aDataB, aLdb, &aBeta, aDataC, aLdc);
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
                           const T &aAlpha, const T *aDataA,
                           const int &aLda, const T &aBeta,
                           T *aDataC, const int &aLdc) {

    auto context = ContextManager::GetOperationContext();
    auto cublas_handle = context->GetCuBlasDnHandle();
    auto triangle = aFillLower ? CUBLAS_FILL_MODE_LOWER
                               : CUBLAS_FILL_MODE_UPPER;
    auto transpose = aTranspose ? CUBLAS_OP_T : CUBLAS_OP_N;

    int rc = 0;

    if constexpr(is_double <T>()) {
        rc = cublasDsyrk(cublas_handle, triangle, transpose, aNumRowA,
                         aNumColA, &aAlpha, aDataA, aLda, &aBeta, aDataC,
                         aLdc);

    } else {
        rc = cublasSsyrk(cublas_handle, triangle, transpose, aNumRowA,
                         aNumColA, &aAlpha, aDataA, aLda, &aBeta, aDataC,
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
                           const T &aAlpha, const T *aDataA,
                           const int &aLda, T *aDataB,
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
                         aNumRowsB, aNumColsB, &aAlpha, aDataA, aLda,
                         aDataB, aLdb);
    } else {
        rc = cublasStrsm(cublas_handle, side, triangle, transpose, diag,
                         aNumRowsB, aNumColsB, &aAlpha, aDataA, aLda,
                         aDataB, aLdb);
    }

    if (rc != 0) {
        MPCR_API_EXCEPTION("Error While Performing Trsm on GPU", rc);
    }
}


template <typename T>
int
GPULinearAlgebra <T>::Potrf(const bool &aFillUpperTri,
                            const int &aNumRow, T *aDataA,
                            const int &aLda) {
    auto triangle = aFillUpperTri ? CUBLAS_FILL_MODE_UPPER
                                  : CUBLAS_FILL_MODE_LOWER;
    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();
    size_t lWork_device = 0;
    size_t lWork_host = 0;
    auto data_type = is_double <T>() ? CUDA_R_64F : CUDA_R_32F;

    cusolverDnXpotrf_bufferSize(cusolver_handle, NULL, triangle, aNumRow,
                                data_type, (void *) aDataA, aNumRow, data_type,
                                &lWork_device, &lWork_host);

    auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);
    auto work_space_host = context->RequestWorkBufferHost(lWork_host);


    cusolverDnXpotrf(cusolver_handle, NULL, triangle, aNumRow, data_type,
                     (void *) aDataA, aNumRow, data_type, work_space_dev,
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
                            const int &aNumRow, T *aDataA,
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
                                    aDataA,
                                    aNumRow,
                                    &lWork);
        auto work_space = context->RequestWorkBufferDevice(lWork * sizeof(T));

        cusolverDnDpotri(cusolver_handle,
                         triangle,
                         aNumRow,
                         aDataA,
                         aNumRow,
                         (T *) work_space,
                         lWork,
                         context->GetInfoPointer());
    } else {
        cusolverDnSpotri_bufferSize(cusolver_handle,
                                    triangle,
                                    aNumRow,
                                    aDataA,
                                    aNumRow,
                                    &lWork);

        auto work_space = context->RequestWorkBufferDevice(lWork * sizeof(T));

        cusolverDnSpotri(cusolver_handle,
                         triangle,
                         aNumRow,
                         aDataA,
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
                           T *aDataA, const int &aLda, void *aIpiv,
                           T *aDataOut, const int &aLdo) {
    // TODO : This function can support half precision internally for the LU factorization
}


template <typename T>
int
GPULinearAlgebra <T>::Getrf(const int &aNumRow, const int &aNumCol,
                            T *aDataA, const int &aLda,
                            int64_t *aIpiv) {

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();

    size_t lWork_device = 0;
    size_t lWork_host = 0;
    auto data_type = is_double <T>() ? CUDA_R_64F : CUDA_R_32F;

    cusolverDnXgetrf_bufferSize(cusolver_handle, NULL, aNumRow, aNumCol,
                                data_type, (void *) aDataA, aLda, data_type,
                                &lWork_device, &lWork_host);

    auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);
    auto work_space_host = context->RequestWorkBufferHost(lWork_host);

    cusolverDnXgetrf(cusolver_handle, NULL, aNumRow, aNumCol, data_type, aDataA,
                     aLda, aIpiv, data_type, work_space_dev, lWork_device,
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
GPULinearAlgebra <T>::Getri(const int &aMatRank, T *aDataA, const int &aLda,
                            int64_t *aIpiv) {

    // NO GPU implementation
}


template <typename T>
int
GPULinearAlgebra <T>::SVD(const signed char &aJob,
                          const int &aNumRow,
                          const int &aNumCol, T *aDataA,
                          const int &aLda, T *aDataS,
                          T *aDataU, const int &aLdu,
                          T *aDataVT, const int &aLdvt) {

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();

    size_t lWork_device = 0;
    size_t lWork_host = 0;
    auto data_type = is_double <T>() ? CUDA_R_64F : CUDA_R_32F;

    cusolverDnXgesvd_bufferSize(cusolver_handle, NULL, aJob, aJob, aNumRow,
                                aNumCol, data_type, (void *) aDataA, aLda,
                                data_type, (void *) aDataS, data_type,
                                (void *) aDataU, aLdu, data_type,
                                (void *) aDataVT, aLdvt, data_type,
                                &lWork_device, &lWork_host);

    auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);
    auto work_space_host = context->RequestWorkBufferHost(lWork_host);

    cusolverDnXgesvd(cusolver_handle, NULL, aJob, aJob, aNumRow, aNumCol,
                     data_type, (void *) aDataA, aLda, data_type,
                     (void *) aDataS, data_type, (void *) aDataU, aLdu,
                     data_type, (void *) aDataVT, aLdvt, data_type,
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
                            const int &aNumCol, T *aDataA,
                            const int64_t &aLda, T *aDataW) {

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
                                data_type, aDataA, aLda, data_type, aDataW,
                                data_type, &lWork_device, &lWork_host);

    auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);
    auto work_space_host = context->RequestWorkBufferHost(lWork_host);

    cusolverDnXsyevd(cusolver_handle, NULL, jobz, triangle, aNumCol, data_type,
                     aDataA, aLda, data_type, aDataW, data_type, work_space_dev,
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
GPULinearAlgebra <T>::Geqp3(const int &aNumRow, const int &aNumCol, T *aDataA,
                            const int &aLda, int64_t *aJpVt, T *aTaw) {

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();

    size_t lWork_device = 0;
    size_t lWork_host = 0;

    auto data_type = is_double <T>() ? CUDA_R_64F : CUDA_R_32F;

    cusolverDnXgeqrf_bufferSize(cusolver_handle, NULL, aNumRow, aNumCol,
                                data_type, aDataA, aLda, data_type, aTaw,
                                data_type, &lWork_device, &lWork_host);

    auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);
    auto work_space_host = context->RequestWorkBufferHost(lWork_host);

    cusolverDnXgeqrf(cusolver_handle, NULL, aNumRow, aNumCol, data_type, aDataA,
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
                            const int &aNumCol, T *aDataA,
                            const int &aLda, const T *aTau) {

    auto context = ContextManager::GetOperationContext();
    auto cusolver_handle = context->GetCusolverDnHandle();
    int lWork_device = 0;

    if constexpr(is_double <T>()) {
        cusolverDnDorgqr_bufferSize(cusolver_handle, aNumRow, aNum, aNumCol,
                                    aDataA, aLda, aTau, &lWork_device);

        auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);

        cusolverDnDorgqr(cusolver_handle, aNumRow, aNum, aNumCol, aDataA, aLda,
                         aTau, (double *) work_space_dev, lWork_device,
                         context->GetInfoPointer());
    } else {
        cusolverDnSorgqr_bufferSize(cusolver_handle, aNumRow, aNum, aNumCol,
                                    aDataA, aLda, aTau, &lWork_device);

        auto work_space_dev = context->RequestWorkBufferDevice(lWork_device);

        cusolverDnSorgqr(cusolver_handle, aNumRow, aNum, aNumCol, aDataA, aLda,
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
                            const T *aData, const int &aLda, T aNormVal,
                            T *aRCond) {
    // NO GPU Implementation
}


template <typename T>
int GPULinearAlgebra <T>::Trcon(const std::string &aNorm,
                                const bool &aUpperTriangle,
                                const bool &aUnitTriangle, const int &aMatOrder,
                                const T *aData, const int &aLda, T *aRCond) {
    // NO GPU Implementation
}
