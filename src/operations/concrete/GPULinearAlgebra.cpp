

#include <operations/concrete/GPULinearAlgerba.hpp>


using namespace mpcr::operations::linear;


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

}


template <typename T>
int
GPULinearAlgebra <T>::Potrf(const bool &aFillUpperTri,
                            const int &aNumRow, T *aDataA,
                            const int &aLda) {

}


template <typename T>
int
GPULinearAlgebra <T>::Potri(const bool &aFillUpperTri,
                            const int &aNumRow, T *aDataA,
                            const int &aLda) {

}


template <typename T>
int
GPULinearAlgebra <T>::Gesv(const int &aNumN, const int &aNumNRH,
                           T *aDataA, const int &aLda, void *aIpiv,
                           T *aDataOut, const int &aLdo) {

}


template <typename T>
int
GPULinearAlgebra <T>::Getrf(const int &aNumRow, const int &aNumCol,
                            T *aDataA, const int &aLda,
                            int64_t *aIpiv) {

}


template <typename T>
int
GPULinearAlgebra <T>::Getri(const int &aMatRank,T *aDataA,const int &aLda, int64_t *aIpiv) {

}


template <typename T>
int
GPULinearAlgebra <T>::SVD(const signed char &aJob,
                          const int &aNumRow,
                          const int &aNumCol, T *aDataA,
                          const int &aLda, T *aDataS,
                          T *aDataU, const int &aLdu,
                          T *aDataVT, const int &aLdvt) {

}


template <typename T>
int
GPULinearAlgebra <T>::Syevd(const bool &aJobzNoVec,
                            const bool &aFillUpperTri,
                            const int &aNumCol, T *aDataA,
                            const int64_t &aLda, T *aDataW) {

}


template <typename T>
int
GPULinearAlgebra <T>::Geqp3(const int &aNumRow, const int &aNumCol, T *aDataA,
                            const int &aLda, int64_t *aJpVt, T *aTaw) {

}


template <typename T>
int
GPULinearAlgebra <T>::Orgqr(const int &aNumRow, const int &aNum,
                            const int &aNumCol, T *aDataA,
                            const int &aLda, const T *aTau) {

}


template <typename T>
int
GPULinearAlgebra <T>::Gecon(const std::string &aNorm, const int &aNumRow,
                            const T *aData, const int &aLda, T aNormVal,
                            T *aRCond) {

}


template <typename T>
int GPULinearAlgebra <T>::Trcon(const std::string &aNorm,
                                const bool &aUpperTriangle,
                                const bool &aUnitTriangle, const int &aMatOrder,
                                const T *aData, const int &aLda, T *aRCond) {

}
