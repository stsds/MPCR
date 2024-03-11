

#include <operations/concrete/CPULinearAlgebra.hpp>
#include <blas.hh>
#include <lapack.hh>


using namespace mpcr::operations::linear;


template <typename T>
void
CPULinearAlgebra <T>::Gemm(const bool &aTransposeA,
                           const bool &aTransposeB,
                           const int &aNumRowsA,
                           const int &aNumColB,
                           const int &aNumRowB,
                           const T &aAlpha, const T *aDataA,
                           const int &aLda, const T *aDataB,
                           const int &aLdb, const T &aBeta,
                           T *aDataC, const int &aLdc) {

    auto transpose_a = aTransposeA ? blas::Op::Trans : blas::Op::NoTrans;
    auto transpose_b = aTransposeB ? blas::Op::Trans : blas::Op::NoTrans;
    auto layout = blas::Layout::ColMajor;

    blas::gemm(layout, transpose_a, transpose_b, aNumRowsA, aNumColB, aNumRowB,
               aAlpha, aDataA, aLda, aDataB, aLdb, aBeta, aDataC, aLdc);

}


template <typename T>
void
CPULinearAlgebra <T>::Syrk(const bool &aFillLower,
                           const bool &aTranspose,
                           const int &aNumRowA,
                           const int &aNumColA,
                           const T &aAlpha, const T *aDataA,
                           const int &aLda, const T &aBeta,
                           T *aDataC, const int &aLdc) {

    auto transpose_a = aTranspose ? blas::Op::Trans : blas::Op::NoTrans;
    auto fill_mode = aFillLower ? blas::Uplo::Lower : blas::Uplo::Upper;
    auto layout = blas::Layout::ColMajor;


    blas::syrk(layout, fill_mode, transpose_a, aNumRowA, aNumColA,
               aAlpha, aDataA, aLda, aBeta, aDataC, aLdc);
}


template <typename T>
void
CPULinearAlgebra <T>::Trsm(const bool &aLeftSide,
                           const bool &aFillUpperTri,
                           const bool &aTranspose,
                           const int &aNumRowsB,
                           const int &aNumColsB,
                           const T &aAlpha, const T *aDataA,
                           const int &aLda, T *aDataB,
                           const int &aLdb) {

    auto side = aLeftSide ? blas::Side::Left : blas::Side::Right;
    auto which_triangle = aFillUpperTri ? blas::Uplo::Upper : blas::Uplo::Lower;
    auto transpose = aTranspose ? blas::Op::Trans : blas::Op::NoTrans;
    auto layout = blas::Layout::ColMajor;

    blas::trsm(layout, side, which_triangle, transpose,
               blas::Diag::NonUnit, aNumRowsB, aNumColsB, aAlpha, aDataA, aLda,
               aDataB, aLdb);
}


template <typename T>
int
CPULinearAlgebra <T>::Potrf(const bool &aFillUpperTri,
                            const int &aNumRow, T *aDataA,
                            const int &aLda) {

    auto triangle = aFillUpperTri ? lapack::Uplo::Upper : lapack::Uplo::Lower;
    auto rc = lapack::potrf(triangle, aNumRow, aDataA, aLda);

    return rc;
}


template <typename T>
int
CPULinearAlgebra <T>::Potri(const bool &aFillUpperTri,
                            const int &aNumRow, T *aDataA,
                            const int &aLda) {

    auto triangle = aFillUpperTri ? lapack::Uplo::Upper : lapack::Uplo::Lower;
    auto rc = lapack::potri(triangle, aNumRow, aDataA, aLda);

    return rc;
}


template <typename T>
int
CPULinearAlgebra <T>::Gesv(const int &aNumN, const int &aNumNRH,
                           T *aDataA, const int &aLda, void *aIpiv,
                           T *aDataB, const int &aLdb, T *aDataOut,
                           const int &aLdo) {

    auto rc = lapack::gesv(aNumN, aNumNRH, aDataA, aLda, (int64_t *) aIpiv,
                           aDataOut, aLdo);

    return rc;

}


template <typename T>
int
CPULinearAlgebra <T>::Getrf(const int &aNumRow, const int &aNumCol,
                            T *aDataA, const int &aLda,
                            int64_t *aIpiv) {

    auto rc = lapack::getrf(aNumRow, aNumCol, aDataA, aLda, aIpiv);
    return rc;
}


template <typename T>
int
CPULinearAlgebra <T>::Getri(const int &aMatRank, T *aDataA, const int &aLda,
                            int64_t *aIpiv) {

    auto rc = lapack::getri(aMatRank, aDataA, aLda, aIpiv);
    return rc;
}


template <typename T>
int
CPULinearAlgebra <T>::SVD(const signed char &aJob,
                          const int &aNumRow,
                          const int &aNumCol, T *aDataA,
                          const int &aLda, T *aDataS,
                          T *aDataU, const int &aLdu,
                          T *aDataVT, const int &aLdvt) {
    lapack::Job job;

    if (aJob == 'N') {
        job = lapack::Job::NoVec;
    } else if (aJob == 'S') {
        job = lapack::Job::SomeVec;
    } else {
        job = lapack::Job::AllVec;
    }

    auto rc = lapack::gesdd(job, aNumRow, aNumCol, aDataA, aLda, aDataS,
                            aDataU, aLdu, aDataVT, aLdvt);

    return rc;

}


template <typename T>
int
CPULinearAlgebra <T>::Syevd(const bool &aJobzNoVec,
                            const bool &aFillUpperTri,
                            const int &aNumCol, T *aDataA,
                            const int64_t &aLda, T *aDataW) {

    auto jobz = aJobzNoVec ? lapack::Job::NoVec : lapack::Job::Vec;
    auto triangle = aFillUpperTri ? lapack::Uplo::Upper : lapack::Uplo::Lower;

    auto rc = lapack::syevd(jobz, triangle, aNumCol, aDataA, aNumCol, aDataW);
    return rc;

}


template <typename T>
int
CPULinearAlgebra <T>::Geqp3(const int &aNumRow, const int &aNumCol, T *aDataA,
                            const int &aLda, int64_t *aJpVt, T *aTaw) {

    auto rc = lapack::geqp3(aNumRow, aNumCol, aDataA, aLda, aJpVt, aTaw);
    return rc;

}


template <typename T>
int
CPULinearAlgebra <T>::Orgqr(const int &aNumRow, const int &aNum,
                            const int &aNumCol, T *aDataA,
                            const int &aLda, const T *aTau) {

    auto rc = lapack::orgqr(aNumRow, aNum, aNumCol, aDataA, aLda, aTau);
    return rc;
}


template <typename T>
int
CPULinearAlgebra <T>::Gecon(const std::string &aNorm, const int &aNumRow,
                            const T *aData, const int &aLda, T aNormVal,
                            T *aRCond) {
    auto norm = aNorm == "I" ? lapack::Norm::Inf : lapack::Norm::One;

    auto rc = lapack::gecon(norm, aNumRow, aData, aLda, aNormVal, aRCond);
    return rc;

}


template <typename T>
int CPULinearAlgebra <T>::Trcon(const std::string &aNorm,
                                const bool &aUpperTriangle,
                                const bool &aUnitTriangle, const int &aMatOrder,
                                const T *aData, const int &aLda, T *aRCond) {
    auto norm = aNorm == "I" ? lapack::Norm::Inf : lapack::Norm::One;
    auto triangle = aUpperTriangle ? lapack::Uplo::Upper : lapack::Uplo::Lower;
    auto diag_unit = aUnitTriangle ? lapack::Diag::Unit : lapack::Diag::NonUnit;

    auto rc = lapack::trcon(norm, triangle, diag_unit, aMatOrder, aData, aLda,
                            aRCond);
    return rc;
}
