
#ifndef MPCR_CPULINEARALGEBRA_HPP
#define MPCR_CPULINEARALGEBRA_HPP

#include <operations/interface/LinearAlgebraBackend.hpp>


namespace mpcr {
    namespace operations {
        namespace linear {
            template <typename T>
            class CPULinearAlgebra : public LinearAlgebraBackend <T> {

            public:

                CPULinearAlgebra() = default;

                ~CPULinearAlgebra() = default;

                void
                Gemm(const bool &aTransposeA, const bool &aTransposeB,
                     const int &aNumRowsA, const int &aNumColB,
                     const int &aNumRowB, const T &aAlpha, const T *aDataA,
                     const int &aLda, const T *aDataB, const int &aLdb,
                     const T &aBeta, T *aDataC, const int &aLdc);

                void
                Syrk(const bool &aFillLower, const bool &aTranspose,
                     const int &aNumRowA, const int &aNumColA, const T &aAlpha,
                     const T *aDataA, const int &aLda, const T &aBeta,
                     T *aDataC, const int &aLdc);


                void
                Trsm(const bool &aLeftSide, const bool &aFillUpperTri,
                     const bool &aTranspose, const int &aNumRowsB,
                     const int &aNumColsB, const T &aAlpha, const T *aDataA,
                     const int &aLda, T *aDataB, const int &aLdb);


                int
                Potrf(const bool &aFillUpperTri, const int &aNumRow, T *aDataA,
                      const int &aLda);


                int
                Potri(const bool &aFillUpperTri, const int &aNumRow, T *aDataA,
                      const int &aLda);


                int
                Gesv(const int &aNumN, const int &aNumNRH, T *aDataA,
                     const int &aLda, void *aIpiv, T *aDataB, const int &aLdb,
                     T *aDataOut, const int &aLdo);

                int
                Getrf(const int &aNumRow, const int &aNumCol, T *aDataA,
                      const int &aLda, int64_t *aIpiv);


                int
                Getri(const int &aMatRank, T *aDataA, const int &aLda,
                      int64_t *aIpiv);


                int
                SVD(const signed char &aJob, const int &aNumRow,
                    const int &aNumCol, T *aDataA, const int &aLda, T *aDataS,
                    T *aDataU, const int &aLdu, T *aDataVT,
                    const int &aLdvt);

                int
                Syevd(const bool &aJobzNoVec, const bool &aFillUpperTri,
                      const int &aNumCol, T *aDataA, const int64_t &aLda,
                      T *aDataW);

                int
                Geqp3(const int &aNumRow, const int &aNumCol, T *aDataA,
                      const int &aLda, int64_t *aJpVt, T *aTaw);

                int
                Orgqr(const int &aNumRow, const int &aNum, const int &aNumCol,
                      T *aDataA, const int &aLda, const T *aTau);

                int
                Gecon(const std::string &aNorm, const int &aNumRow,
                      const T *aData, const int &aLda, T aNormVal, T *aRCond);

                int
                Trcon(const std::string &aNorm, const bool &aUpperTriangle,
                      const bool &aUnitTriangle, const int &aMatOrder,
                      const T *aData,
                      const int &aLda, T *aRCond);
            };

            MPCR_INSTANTIATE_CLASS(CPULinearAlgebra)

        }
    }
}

#endif //MPCR_CPULINEARALGEBRA_HPP
