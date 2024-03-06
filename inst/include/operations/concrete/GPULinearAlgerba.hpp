

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
                     const int &aNumRowB, const T &aAlpha, const T *aDataA,
                     const int &aLda, const T *aDataB, const int &aLdb,
                     const T &aBeta, T *aDataC, const int &aLdc);

                // crossprod - blas - avail



                void
                Syrk(const bool &aFillLower, const bool &aTranspose,
                     const int &aNumRowA, const int &aNumColA, const T &aAlpha,
                     const T *aDataA, const int &aLda, const T &aBeta,
                     T *aDataC, const int &aLdc);


                //backsolve - blas - avail
                // missing diag in args
                void
                Trsm(const bool &aLeftSide, const bool &aFillUpperTri,
                     const bool &aTranspose, const int &aNumRowsB,
                     const int &aNumColsB, const T &aAlpha, const T *aDataA,
                     const int &aLda, T *aDataB, const int &aLdb);

                //chol - lapack - avail
                int
                Potrf(const bool &aFillUpperTri, const int &aNumRow, T *aDataA,
                      const int &aLda);

                //chol inv - lapack - avail
                int
                Potri(const bool &aFillUpperTri, const int &aNumRow, T *aDataA,
                      const int &aLda);

                // solve - lapack - avail
                // one is taking int and the other is int64_t
                // might need refactor if both doesn't match the same format.
                int
                Gesv(const int &aNumN, const int &aNumNRH, T *aDataA,
                     const int &aLda, void *aIpiv, T *aDataOut,
                     const int &aLdo);

                //solve - lapack - avail
                // might need revision since 2 more params are added.
                int
                Getrf(const int &aNumRow, const int &aNumCol, T *aDataA,
                      const int &aLda, int64_t *aIpiv);

                //solve - lapack -> no direct call
                int
                Getri(const int &aMatRank,T *aDataA,const int &aLda, int64_t *aIpiv);


                //svd - lapack  -> gesvd (GPU) and gesdd(CPU)
                int
                SVD(const signed char &aJob, const int &aNumRow,
                    const int &aNumCol, T *aDataA, const int &aLda, T *aDataS,
                    T *aDataU, const int &aLdu, T *aDataVT,
                    const int &aLdvt);

                //eigen - lapack -> syevd (GPU)
                int
                Syevd(const bool &aJobzNoVec, const bool &aFillUpperTri,
                      const int &aNumCol, T *aDataA, const int64_t &aLda,
                      T *aDataW);

                //QR - lapack -> geqrf
                int
                Geqp3(const int &aNumRow, const int &aNumCol, T *aDataA,
                      const int &aLda, int64_t *aJpVt, T *aTaw);

                //QR-QY
                // QR-Q - lapack - avail
                int
                Orgqr(const int &aNumRow, const int &aNum, const int &aNumCol,
                      T *aDataA, const int &aLda, const T *aTau);

                //RCond - lapack -> no direct call
                int
                Gecon(const std::string &aNorm, const int &aNumRow,
                      const T *aData, const int &aLda, T aNormVal, T *aRCond);

                int
                Trcon(const std::string &aNorm, const bool &aUpperTriangle,
                      const bool &aUnitTriangle, const int &aMatOrder,const T *aData,
                      const int &aLda,T *aRCond);

            };

            MPCR_INSTANTIATE_CLASS(GPULinearAlgebra)
        }
    }
}
#endif //MPCR_GPULINEARALGERBA_HPP
