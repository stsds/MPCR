

#ifndef MPR_RLINEARALGEBRA_HPP
#define MPR_RLINEARALGEBRA_HPP


#include <operations/LinearAlgebra.hpp>


/**
 * @brief
 * Calculate CrossProduct of 2 MPR Matrices
 * performs:
 * x %*% y  , t(x) %*% x
 *
 * @param[in] aInputA
 * MPR Matrix
 * @param[in] aInputB
 * MPR Matrix, if Null t(aInputA) %*% aInputA ,otherwise aInputA %*% aInputB
 * @returns
 * MPR Matrix
 *
 */
DataType *
RCrossProduct(DataType *aInputA, SEXP aInputB);

/**
 * @brief
 * Calculate CrossProduct of 2 MPR Matrices
 * performs:
 * x %*% t(y)  , x %*% t(x)
 *
 * @param[in] aInputA
 * MPR Matrix
 * @param[in] aInputB
 * MPR Matrix, if Null aInputA %*% t(aInputA) ,otherwise aInputA %*% t(aInputB)
 * @returns
 * MPR Matrix
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
 * MPR Square Matrix
 * @param[in] aOnlyValues
 * bool True, Only values will be returned ,otherwise values and vectors.
 * @returns
 * vector of MPR objects, First element Values ,and second element Vectors.
 */
std::vector <DataType>
REigen(DataType *aInputA, const bool &aOnlyValues);

/**
 * @brief
 * Check if a Matrix Is Symmetric
 *
 * @param[in] aInput
 * MPR Matrix
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
 * MPR Matrix
 * @param[in] aInputB
 * MPR Matrix
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
 * MPR Matrix
 * @returns
 * MPR Matrix containing decomposition result
 *
 */
DataType *
RCholesky(DataType *aInputA);

/**
 * @brief
 * Invert a symmetric, positive definite square matrix from its
 * Cholesky decomposition.
 *
 * @param[in] aInput
 * MPR Matrix containing Cholesky decomposition.
 * @returns
 * MPR Matrix
 *
 */
DataType *
RCholeskyInv(DataType *aInputA, const size_t &aSize);

/**
 * @brief
 * Solves the equation AX=B
 *
 * @param[in] aInputA
 * MPR Matrix A
 * @param[in] aInputB
 * MPR Matrix X, if Null t(A) will be used.
 * @returns
 * MPR Matrix B
 *
 */
DataType *
RSolve(DataType *aInputA, SEXP aInputB);

/**
 * @brief
 * Compute the singular-value decomposition of a rectangular matrix.
 *
 * @param[in] aInputA
 * MPR Matrix
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
std::vector <DataType >
RSVD(DataType *aInputA, const long &aNu, const long &aNv,
     const bool &aTranspose);

/**
 * @brief
 * Transpose a given MPR Matrix
 *
 * @param[in,out] aInputA
 * MPR Matrix A
 *
 */
DataType*
RTranspose(DataType *aInputA);

/**
 * @brief
 * Computes a matrix norm of aInput. The norm can be the one ("O") norm,
 * the infinity ("I") norm, the Frobenius ("F") norm, or the maximum
 * modulus ("M") among elements of a matrix.
 *
 * @param[in] aInput
 * MPR Matrix
 * @param[in] aType
 * Type of Norm ( O , 1 , I , F, M)
 * @returns
 * Norm Value
 */
DataType *
RNorm(DataType *aInputA, const std::string &aType);

/**
 * @brief
 * Computes the QR decomposition of a matrix.
 *
 * @param[in] aInputA
 * MPR Matrix
 * @returns
 * vector containing QR,QRaux,Pivot,Rank
 *
 */
std::vector <DataType >
RQRDecomposition(DataType *aInputA);

/**
 * @brief
 * Estimate the reciprocal of the condition number of a matrix.
 *
 * @param[in] aInputA
 * MPR Matrix
 * @param[in] aNorm
 * character string indicating the type of norm to be used in the
 * estimate. The default is "O" for the 1-norm ("O" is equivalent to "1").
 * For sparse matrices,  when useInv=TRUE, norm can be any of the kinds allowed for norm;
 * otherwise, the other possible value is "I" for the infinity norm.
 * @param[in] aTriangle
 * Bool if true,Only the lower triangle will be used.
 * @returns
 * MPR Vector containing one element which is an estimate of the
 * reciprocal condition number of aInput.
 *
 */
DataType *
RRCond(DataType *aInputA, const std::string &aNorm, const bool &aTriangle);

/**
 * @brief
 * returns R. This may be pivoted,
 * The number of rows of R is either nrow(aInputA) or ncol(aInputA)
 * (and may depend on whether complete is TRUE or FALSE).
 *
 * @param[in] aInputA
 * MPR Matrix
 * @param[in] aComplete
 * logical expression . Indicates whether an arbitrary
 * orthogonal completion of the Q or X matrices is to be made,
 * or whether the  matrix is to be completed by binding zero-value
 * rows beneath the square upper triangle.
 * @returns
 * returns R. This may be pivoted. As MPR Object.
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
 * MPR Matrix QR
 * @param[in] aInputB
 * MPR Object Representing QRAUX
 * @param[in] aComplete
 * logical expression . Indicates whether an arbitrary
 * orthogonal completion of the Q or X matrices is to be made,
 * or whether the  matrix is to be completed by binding zero-value
 * rows beneath the square upper triangle.
 * @param[in] aDvec
 * MPR Object Representing DVec ,if null QRDecompositionQ will be called.
 * otherwise QRDecompositionQY
 * @returns
 * returns Q. As MPR Object.
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
 * MPR Matrix QR
 * @param[in] aInputB
 * MPR Object Representing QRAUX
 * @param[in] aComplete
 * logical expression . Indicates whether an arbitrary
 * orthogonal completion of the Q or X matrices is to be made,
 * or whether the  matrix is to be completed by binding zero-value
 * rows beneath the square upper triangle.
 * @param[in] aDvec
 * MPR Object Representing DVec , QRDecompositionQY with flag transpose =false
 * wil be called
 * @returns
 * returns Q. As MPR Object.
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
 * MPR Matrix QR
 * @param[in] aInputB
 * MPR Object Representing QRAUX
 * @param[in] aComplete
 * logical expression . Indicates whether an arbitrary
 * orthogonal completion of the Q or X matrices is to be made,
 * or whether the  matrix is to be completed by binding zero-value
 * rows beneath the square upper triangle.
 * @param[in] aDvec
 * MPR Object Representing DVec , QRDecompositionQY with flag transpose =true
 * wil be called
 * @returns
 * returns Q. As MPR Object.
 *
 */
DataType *
RQRDecompositionQty(DataType *aInputA, DataType *aInputB, DataType *aDvec);


#endif //MPR_RLINEARALGEBRA_HPP
