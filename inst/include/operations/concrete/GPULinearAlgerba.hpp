

#ifndef MPCR_GPULINEARALGERBA_HPP
#define MPCR_GPULINEARALGERBA_HPP

#include <operations/interface/LinearAlgebraBackend.hpp>


namespace mpcr {
    namespace operations {
        namespace linear {
            template <typename T>
            class GPULinearAlgebra : public LinearAlgebraBackend <T> {

            public:

                GPULinearAlgebra() = default;

                ~GPULinearAlgebra() = default;


                void
                Gemm(const bool &aTransposeA, const bool &aTransposeB,
                     const int &aNumRowsA, const int &aNumColB,
                     const int &aNumRowB, const T &aAlpha, const T *apDataA,
                     const int &aLda, const T *apDataB, const int &aLdb,
                     const T &aBeta, T *apDataC, const int &aLdc);

                // crossprod - blas - avail



                void
                Syrk(const bool &aFillLower, const bool &aTranspose,
                     const int &aNumRowA, const int &aNumColA, const T &aAlpha,
                     const T *apDataA, const int &aLda, const T &aBeta,
                     T *apDataC, const int &aLdc);


                //backsolve - blas - avail
                // missing diag in args
                void
                Trsm(const bool &aLeftSide, const bool &aFillUpperTri,
                     const bool &aTranspose, const int &aNumRowsB,
                     const int &aNumColsB, const T &aAlpha, const T *apDataA,
                     const int &aLda, T *apDataB, const int &aLdb);

                //chol - lapack - avail
                int
                Potrf(const bool &aFillUpperTri, const int &aNumRow, T *apDataA,
                      const int &aLda);

                //chol inv - lapack - avail
                int
                Potri(const bool &aFillUpperTri, const int &aNumRow, T *apDataA,
                      const int &aLda);

                // solve - lapack - avail
                // one is taking int and the other is int64_t
                // might need refactor if both doesn't match the same format.
                int
                Gesv(const int &aNumN, const int &aNumNRH, T *apDataA,
                     const int &aLda, void *apIpiv,  T *apDataB,
                     const int &aLdb,T* apDataOut,const int &aLdo);

                //solve - lapack - avail
                // might need revision since 2 more params are added.
                int
                Getrf(const int &aNumRow, const int &aNumCol, T *apDataA,
                      const int &aLda, int64_t *apIpiv);

                //solve - lapack -> no direct call
                int
                Getri(const int &aMatRank,T *apDataA,const int &aLda, int64_t *apIpiv);


                //svd - lapack  -> gesvd (GPU) and gesdd(CPU)
                int
                SVD(const signed char &aJob, const int &aNumRow,
                    const int &aNumCol, T *apDataA, const int &aLda, T *apDataS,
                    T *apDataU, const int &aLdu, T *apDataVT,
                    const int &aLdvt);

                //eigen - lapack -> syevd (GPU)
                int
                Syevd(const bool &aJobzNoVec, const bool &aFillUpperTri,
                      const int &aNumCol, T *apDataA, const int64_t &aLda,
                      T *apDataW);

                //QR - lapack -> geqrf
                int
                Geqp3(const int &aNumRow, const int &aNumCol, T *apDataA,
                      const int &aLda, int64_t *aJpVt, T *aTaw);

                //QR-QY
                // QR-Q - lapack - avail
                int
                Orgqr(const int &aNumRow, const int &aNum, const int &aNumCol,
                      T *apDataA, const int &aLda, const T *aTau);

                //RCond - lapack -> no direct call
                int
                Gecon(const std::string &aNorm, const int &aNumRow,
                      const T *apData, const int &aLda, T aNormVal, T *aRCond);

                int
                Trcon(const std::string &aNorm, const bool &aUpperTriangle,
                      const bool &aUnitTriangle, const int &aMatOrder,const T *apData,
                      const int &aLda,T *aRCond);


                int
                Getrs(const bool &aTransposeA, const size_t &aNumRowA,
                      const size_t &aNumRhs, const T *apDataA,
                      const size_t &aLda, const int64_t *apIpiv, T *apDataB,
                      const size_t &aLdb);

            };

            MPCR_INSTANTIATE_CLASS(GPULinearAlgebra)
        }
    }
}
#endif //MPCR_GPULINEARALGERBA_HPP
