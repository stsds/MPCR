

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


                void
                Syrk(const bool &aFillLower, const bool &aTranspose,
                     const int &aNumRowA, const int &aNumColA, const T &aAlpha,
                     const T *apDataA, const int &aLda, const T &aBeta,
                     T *apDataC, const int &aLdc);


                void
                Trsm(const bool &aLeftSide, const bool &aFillUpperTri,
                     const bool &aTranspose, const int &aNumRowsB,
                     const int &aNumColsB, const T &aAlpha, const T *apDataA,
                     const int &aLda, T *apDataB, const int &aLdb);


                int
                Potrf(const bool &aFillUpperTri, const int &aNumRow, T *apDataA,
                      const int &aLda);


                int
                Potri(const bool &aFillUpperTri, const int &aNumRow, T *apDataA,
                      const int &aLda);

                int
                Gesv(const int &aNumN, const int &aNumNRH, T *apDataA,
                     const int &aLda, void *apIpiv, T *apDataB,
                     const int &aLdb, T *apDataOut, const int &aLdo,
                     const std::string &aInternalPrecision = "same");


                int
                Getrf(const int &aNumRow, const int &aNumCol, T *apDataA,
                      const int &aLda, int64_t *apIpiv);

                int
                Getri(const int &aMatRank, T *apDataA, const int &aLda,
                      int64_t *apIpiv);

                int
                SVD(const signed char &aJob, const int &aNumRow,
                    const int &aNumCol, T *apDataA, const int &aLda, T *apDataS,
                    T *apDataU, const int &aLdu, T *apDataVT,
                    const int &aLdvt);

                int
                Syevd(const bool &aJobzNoVec, const bool &aFillUpperTri,
                      const int &aNumCol, T *apDataA, const int64_t &aLda,
                      T *apDataW);

                int
                Geqp3(const int &aNumRow, const int &aNumCol, T *apDataA,
                      const int &aLda, int64_t *aJpVt, T *aTaw);

                int
                Orgqr(const int &aNumRow, const int &aNum, const int &aNumCol,
                      T *apDataA, const int &aLda, const T *aTau);

                int
                Gecon(const std::string &aNorm, const int &aNumRow,
                      const T *apData, const int &aLda, T aNormVal, T *aRCond);

                int
                Trcon(const std::string &aNorm, const bool &aUpperTriangle,
                      const bool &aUnitTriangle, const int &aMatOrder,
                      const T *apData,
                      const int &aLda, T *aRCond);


                int
                Getrs(const bool &aTransposeA, const size_t &aNumRowA,
                      const size_t &aNumRhs, const T *apDataA,
                      const size_t &aLda, const int64_t *apIpiv, T *apDataB,
                      const size_t &aLdb);


                int
                Trtri(const size_t &aSideLength, T *apDataA, const size_t &aLda,
                      const bool &aUpperTri);


#ifdef USING_HALF

                void
                HalfGemm(const bool &aTransposeA, const bool &aTransposeB,
                         const int &aNumRowsA, const int &aNumColB,
                         const int &aNumRowB, const half &aAlpha,
                         const half *apDataA,
                         const int &aLda, const half *apDataB, const int &aLdb,
                         const half &aBeta, half *apDataC, const int &aLdc);

#endif


            };

            MPCR_INSTANTIATE_CLASS(GPULinearAlgebra)
        }
    }
}
#endif //MPCR_GPULINEARALGERBA_HPP
