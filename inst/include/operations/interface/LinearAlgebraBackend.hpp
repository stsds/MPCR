
#ifndef MPCR_LINEARALGEBRABACKEND_HPP
#define MPCR_LINEARALGEBRABACKEND_HPP

#include <inttypes.h>
#include <utilities/MPCRDispatcher.hpp>


namespace mpcr {
    namespace operations {
        namespace linear {
            template <typename T>
            class LinearAlgebraBackend {

            public:

                LinearAlgebraBackend() = default;

                ~LinearAlgebraBackend() = default;
                //crossprod -blas - avail


                virtual
                void
                Gemm(const bool &aTransposeA, const bool &aTransposeB,
                     const int &aNumRowsA, const int &aNumColB,
                     const int &aNumRowB, const T &aAlpha, const T *aDataA,
                     const int &aLda, const T *aDataB, const int &aLdb,
                     const T &aBeta, T *aDataC, const int &aLdc) = 0;

                // crossprod - blas - avail

                virtual
                void
                Syrk(const bool &aFillLower, const bool &aTranspose,
                     const int &aNumRowA, const int &aNumColA, const T &aAlpha,
                     const T *aDataA, const int &aLda, const T &aBeta,
                     T *aDataC, const int &aLdc) = 0;

                //backsolve - blas - avail
                // missing diag in args
                virtual
                void
                Trsm(const bool &aLeftSide, const bool &aFillUpperTri,
                     const bool &aTranspose, const int &aNumRowsB,
                     const int &aNumColsB, const T &aAlpha, const T *aDataA,
                     const int &aLda, T *aDataB, const int &aLdb) = 0;

                //chol - lapack - avail
                virtual
                int
                Potrf(const bool &aFillUpperTri, const int &aNumRow, T *aDataA,
                      const int &aLda) = 0;

                //chol inv - lapack - avail
                virtual
                int
                Potri(const bool &aFillUpperTri, const int &aNumRow, T *aDataA,
                      const int &aLda) = 0;

                // solve - lapack - avail
                // one is taking int and the other is int64_t
                // might need refactor if both doesn't match the same format.
                virtual
                int
                Gesv(const int &aNumN, const int &aNumNRH, T *aDataA,
                     const int &aLda, void *aIpiv, T *aDataOut,
                     const int &aLdo) = 0;

                //solve - lapack - avail
                // might need revision since 2 more params are added.
                virtual
                int
                Getrf(const int &aNumRow, const int &aNumCol, T *aDataA,
                      const int &aLda, int64_t *aIpiv) = 0;

                //solve - lapack -> no direct call
                virtual
                int
                Getri(const int &aMatRank,T *aDataA,const int &aLda, int64_t *aIpiv) = 0;


                //svd - lapack  -> gesvd (GPU) and gesdd(CPU)
                virtual
                int
                SVD(const signed char &aJob, const int &aNumRow,
                    const int &aNumCol, T *aDataA, const int &aLda, T *aDataS,
                    T *aDataU, const int &aLdu, T *aDataVT,
                    const int &aLdvt) = 0;

                //eigen - lapack -> syevd (GPU) syevr(CPU)
                virtual
                int
                Syevd(const bool &aJobzNoVec, const bool &aFillUpperTri,
                      const int &aNumCol, T *aDataA, const int64_t &aLda,
                      T *aDataW) = 0;

                //QR - lapack -> geqrf
                virtual
                int
                Geqp3(const int &aNumRow, const int &aNumCol, T *aDataA,
                      const int &aLda, int64_t *aJpVt, T *aTaw) = 0;

                //QR-QY
                // QR-Q - lapack - avail
                virtual
                int
                Orgqr(const int &aNumRow, const int &aNum, const int &aNumCol,
                      T *aDataA, const int &aLda, const T *aTau) = 0;

                //RCond - lapack -> no direct call
                virtual
                int
                Gecon(const std::string &aNorm, const int &aNumRow,
                      const T *aData, const int &aLda, T aNormVal,
                      T *aRCond) = 0;

                virtual
                int
                Trcon(const std::string &aNorm, const bool &aUpperTriangle,
                      const bool &aUnitTriangle, const int &aMatOrder,
                      const T *aData,const int &aLda, T *aRCond) = 0;


            };

            MPCR_INSTANTIATE_CLASS(LinearAlgebraBackend)
        }
    }
}


#endif //MPCR_LINEARALGEBRABACKEND_HPP
