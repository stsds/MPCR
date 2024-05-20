/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_RLINEARALGEBRA_HPP
#define MPCR_RLINEARALGEBRA_HPP


#include <operations/LinearAlgebra.hpp>


/**
 * @brief
 * Calculate CrossProduct of 2 MPCR Matrices
 * performs:
 * x %*% y  , t(x) %*% x
 *
 * @param[in] aInputA
 * MPCR Matrix
 * @param[in] aInputB
 * MPCR Matrix, if Null t(aInputA) %*% aInputA ,otherwise aInputA %*% aInputB
 * @returns
 * MPCR Matrix
 *
 */
DataType *
RCrossProduct(DataType *aInputA, SEXP aInputB);

/**
 * @brief
 * Calculate CrossProduct of 2 MPCR Matrices
 * performs:
 * x %*% t(y)  , x %*% t(x)
 *
 * @param[in] aInputA
 * MPCR Matrix
 * @param[in] aInputB
 * MPCR Matrix, if Null aInputA %*% t(aInputA) ,otherwise aInputA %*% t(aInputB)
 * @returns
 * MPCR Matrix
 *
 */
DataType *
RTCrossProduct(DataType *aInputA, SEXP aInputB);

/**
 * @brief
 * Calculate Eigen Values and (optionally) Eigen Vectors.
 * if(aOnlyValues)= true , only the values will be calculated
 *
 * @param[in] aInput
 * MPCR Square Matrix
 * @param[in] aOnlyValues
 * bool True, Only values will be returned ,otherwise values and vectors.
 * @returns
 * vector of MPCR objects, First element Values ,and second element Vectors.
 */
std::vector <DataType>
REigen(DataType *aInputA, const bool &aOnlyValues);

/**
 * @brief
 * Check if a Matrix Is Symmetric
 *
 * @param[in] aInput
 * MPCR Matrix
 * @returns
 * true if symmetric ,false otherwise
 *
 */
bool
RIsSymmetric(DataType *aInputA);

/**
 * @brief
 * Solves a system of linear equations where the coefficient matrix
 * is upper or lower triangular.
 * Solve aInputA aOutput = aInputB
 *
 * @param[in] aInputA
 * MPCR Matrix
 * @param[in] aInputB
 * MPCR Matrix
 * @param[in] aCol
 * The number of columns of aInputA and rows of aInputB to use.
 * default ncol(aInputA)
 * @param[in] aUpperTriangle
 * logical; if true (default), the upper triangular part of aInputA
 * is used. Otherwise, the lower one.
 * @param[in] aTranspose
 * logical; if true, solve  for t(aInputA) %*% aOutput == aInputB.
 * @returns
 * The solution of the triangular system
 */
DataType *
RBackSolve(DataType *aInputA, DataType *aInputB, const long &aCol,
           const bool &aUpperTriangle, const bool &aTranspose);

/**
 * @brief
 * Calculate Cholesky decomposition
 *
 * @param[in] aInput
 * MPCR Matrix
 * @returns
 * MPCR Matrix containing decomposition result
 *
 */
DataType *
RCholesky(DataType *aInputA, const bool &aUpperTriangle);

/**
 * @brief
 * Invert a symmetric, positive definite square matrix from its
 * Cholesky decomposition.
 *
 * @param[in] aInput
 * MPCR Matrix containing Cholesky decomposition.
 * @returns
 * MPCR Matrix
 *
 */
DataType *
RCholeskyInv(DataType *aInputA, const size_t &aSize);

/**
 * @brief
 * Solves the equation AX=B
 *
 * @param[in] aInputA
 * MPCR Matrix A
 * @param[in] aInputB
 * MPCR Matrix X, if Null t(A) will be used.
 * @param [in] aInternalPrecision
 * string that indicate the precision used for the LU factorization in
 * case of using GESV routine in CuSolver.
 * Options: (float or single) ,half ,tensorfloat
 *
 * @returns
 * MPCR Matrix B
 *
 */
DataType *
RSolve(DataType *aInputA, SEXP aInputB,const std::string &aInternalPrecision);

/**
 * @brief
 * Compute the singular-value decomposition of a rectangular matrix.
 *
 * @param[in] aInputA
 * MPCR Matrix
 * @param[in] aNu
 * the number of left singular vectors to be computed.
 * This must between 0 and m = nrow(aInputA).
 * default aNu = nrow(aInputA).
 * @param[in] aNv
 * the number of right singular vectors to be computed.
 * This must be between 0 and n = ncol(x).
 * default aNv = ncol(aInputA).
 * @param[in] aTranspose
 * Bool if true, aOutputV will contain V ,otherwise VT
 * default true for svd() false for la.svd()
 * @returns
 * Vectors containing d,u,v or VT if aTranspose = false
 *
 */
std::vector <DataType>
RSVD(DataType *aInputA, const long &aNu, const long &aNv,
     const bool &aTranspose);

/**
 * @brief
 * Transpose a given MPCR Matrix
 *
 * @param[in,out] aInputA
 * MPCR Matrix A
 *
 */
DataType *
RTranspose(DataType *aInputA);

/**
 * @brief
 * Computes a matrix norm of aInput. The norm can be the one ("O") norm,
 * the infinity ("I") norm, the Frobenius ("F") norm, or the maximum
 * modulus ("M") among elements of a matrix.
 *
 * @param[in] aInput
 * MPCR Matrix
 * @param[in] aType
 * Type of Norm ( O , 1 , I , F, M)
 * @returns
 * Norm Value
 */
double
RNorm(DataType *aInputA, const std::string &aType);

/**
 * @brief
 * Computes the QR decomposition of a matrix.
 *
 * @param[in] aInputA
 * MPCR Matrix
 * @param[in] aTolerance
 * the tolerance for detecting linear dependencies in the columns of
 * aInputA
 * @returns
 * vector containing QR,QRaux,Pivot,Rank
 *
 */
std::vector <DataType>
RQRDecomposition(DataType *aInputA,const double &aTolerance);

/**
 * @brief
 * Estimate the reciprocal of the condition number of a matrix.
 *
 * @param[in] aInputA
 * MPCR Matrix
 * @param[in] aNorm
 * character string indicating the type of norm to be used in the
 * estimate. The default is "O" for the 1-norm ("O" is equivalent to "1").
 * For sparse matrices,  when useInv=TRUE, norm can be any of the kinds allowed for norm;
 * otherwise, the other possible value is "I" for the infinity norm.
 * @param[in] aTriangle
 * Bool if true,Only the lower triangle will be used.
 * @returns
 * MPCR Vector containing one element which is an estimate of the
 * reciprocal condition number of aInput.
 *
 */
double
RRCond(DataType *aInputA, const std::string &aNorm, const bool &aTriangle);

/**
 * @brief
 * returns R. This may be pivoted,
 * The number of rows of R is either nrow(aInputA) or ncol(aInputA)
 * (and may depend on whether complete is TRUE or FALSE).
 *
 * @param[in] aInputA
 * MPCR Matrix
 * @param[in] aComplete
 * logical expression . Indicates whether an arbitrary
 * orthogonal completion of the Q or X matrices is to be made,
 * or whether the  matrix is to be completed by binding zero-value
 * rows beneath the square upper triangle.
 * @returns
 * returns R. This may be pivoted. As MPCR Object.
 *
 */
DataType *
RQRDecompositionR(DataType *aInputA, const bool &aComplete);

/**
 * @brief
 * Returns part or all of Q, the order-nrow(X) orthogonal (unitary)
 * transformation represented by qr. If complete is TRUE, Q has
 * nrow(X) columns. If complete is FALSE, Q has ncol(X) columns.
 * and each column of Q is multiplied by the corresponding value in Dvec.
 *
 * @param[in] aInputA
 * MPCR Matrix QR
 * @param[in] aInputB
 * MPCR Object Representing QRAUX
 * @param[in] aComplete
 * logical expression . Indicates whether an arbitrary
 * orthogonal completion of the Q or X matrices is to be made,
 * or whether the  matrix is to be completed by binding zero-value
 * rows beneath the square upper triangle.
 * @param[in] aDvec
 * MPCR Object Representing DVec ,if null QRDecompositionQ will be called.
 * otherwise QRDecompositionQY
 * @returns
 * returns Q. As MPCR Object.
 *
 */
DataType *
RQRDecompositionQ(DataType *aInputA, DataType *aInputB, const bool &aComplete,
                  SEXP aDvec);

/**
 * @brief
 * Returns part or all of Q, the order-nrow(X) orthogonal (unitary)
 * transformation represented by qr. If complete is TRUE, Q has
 * nrow(X) columns. If complete is FALSE, Q has ncol(X) columns.
 * and each column of Q is multiplied by the corresponding value in Dvec.
 *
 * @param[in] aInputA
 * MPCR Matrix QR
 * @param[in] aInputB
 * MPCR Object Representing QRAUX
 * @param[in] aComplete
 * logical expression . Indicates whether an arbitrary
 * orthogonal completion of the Q or X matrices is to be made,
 * or whether the  matrix is to be completed by binding zero-value
 * rows beneath the square upper triangle.
 * @param[in] aDvec
 * MPCR Object Representing DVec , QRDecompositionQY with flag transpose =false
 * wil be called
 * @returns
 * returns Q. As MPCR Object.
 *
 */
DataType *
RQRDecompositionQy(DataType *aInputA, DataType *aInputB, DataType *aDvec);

/**
 * @brief
 * Returns part or all of Q, the order-nrow(X) orthogonal (unitary)
 * transformation represented by qr. If complete is TRUE, Q has
 * nrow(X) columns. If complete is FALSE, Q has ncol(X) columns.
 * and each column of Q is multiplied by the corresponding value in Dvec.
 *
 * @param[in] aInputA
 * MPCR Matrix QR
 * @param[in] aInputB
 * MPCR Object Representing QRAUX
 * @param[in] aComplete
 * logical expression . Indicates whether an arbitrary
 * orthogonal completion of the Q or X matrices is to be made,
 * or whether the  matrix is to be completed by binding zero-value
 * rows beneath the square upper triangle.
 * @param[in] aDvec
 * MPCR Object Representing DVec , QRDecompositionQY with flag transpose =true
 * wil be called
 * @returns
 * returns Q. As MPCR Object.
 *
 */
DataType *
RQRDecompositionQty(DataType *aInputA, DataType *aInputB, DataType *aDvec);

/**
 * @brief
 * Perform Matrix Matrix Multiplication using Gemm/Syrk routines
 * performs:
 * C= alphaA *B +betaC
 *
 * @param[in] aInputA
 * MPCR Matrix
 * @param[in] aInputB
 * MPCR Matrix ,if missing syrk routine will be used
 * @param[in,out] aInputC
 * MPCR Matrix
 * @param[in] aTransposeA
 * if true , the transpose of A will be Used
 * @param[in] aTransposeB
 * if true , the transpose of B will be Used
 * @param aAlpha
 * factor of A
 * @param aBeta
 * factor of C
 *
 * @returns
 * MPCR Matrix
 *
 */
void
RGemm(DataType *aInputA, SEXP aInputB, DataType *aInputC,
      const bool &aTransposeA, const bool &aTransposeB, const double &aAlpha,
      const double &aBeta);

DataType *
RTrsm(DataType *aInputA, DataType *aInputB, const bool &aUpperTri,
      const bool &aTranspose, const char &aSide, const double &aAlpha);


#endif //MPCR_RLINEARALGEBRA_HPP
