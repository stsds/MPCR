
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
                /**
                 * @brief
                 * Default constructor
                 */
                LinearAlgebraBackend() = default;

                /**
                 * @brief
                 * Default destructor
                 */
                ~LinearAlgebraBackend() = default;

                /**
                 * @brief
                 * Computes a matrix-matrix product with general matrices.
                 * Solves :  C = alpha * op(A) * op(B) + beta * C
                 * Where op: op(X)=X or op(X)=X^T
                 *
                 * @param [in] aTransposeA
                 * if True op(A) = A^T ,otherwise op(A)= A
                 * @param [in] aTransposeB
                 * if True op(B) = B^T ,otherwise op(B)= B
                 * @param [in] aNumRowsA
                 * Number of rows in matrix A
                 * @param [in] aNumColB
                 * Number of cols in matrix B
                 * @param [in] aNumRowB
                 * Number of rows in matrix B and cols in matrix A
                 * @param [in] aAlpha
                 * Scalar alpha. If alpha is zero, A and B are not accessed
                 * @param [in] aDataA
                 * Matrix A data
                 * @param [in] aLda
                 * Leading dimension for matrix A
                 * @param [in] aDataB
                 * Matrix B data
                 * @param [in] aLdb
                 * Leading dimension for matrix B
                 * @param [in] aBeta
                 * Scalar beta. If beta is zero, C need not be set on input.
                 * @param [in,out] aDataC
                 * input and output matrix C, will be used as an input in case
                 * aBeta != 0
                 * @param [in] aLdc
                 * Leading dimension of matrix C
                 *
                 */
                virtual
                void
                Gemm(const bool &aTransposeA, const bool &aTransposeB,
                     const int &aNumRowsA, const int &aNumColB,
                     const int &aNumRowB, const T &aAlpha, const T *aDataA,
                     const int &aLda, const T *aDataB, const int &aLdb,
                     const T &aBeta, T *aDataC, const int &aLdc) = 0;

                /**
                 * @brief
                 * Computes a matrix-matrix product with general matrices.
                 * Solves :  C = alpha * A * A^T + beta * C
                 *           C = alpha * A^T * A + beta * C
                 *
                 *
                 * @param [in] aFillLower
                 * What part of the matrix C is referenced,
                 * the opposite triangle being assumed from symmetry
                 * if TRUE, Lower triangle will be used, otherwise upper triangle
                 * @param [in] aTranspose
                 * if True, the operation will be as follow:
                 * C = alpha * A^T * A + beta * C
                 * otherwise, C = alpha * A * A^T + beta * C
                 * @param [in] aNumRowA
                 * Number of rows in matrix A
                 * @param [in] aNumColA
                 * Number of cols in matrix A
                 * @param [in] aAlpha
                 * Scalar alpha. If alpha is zero, A is not accessed
                 * @param [in] aDataA
                 * Matrix A data
                 * @param [in] aLda
                 * Leading dimension for matrix A
                 * @param [in] aBeta
                 * Scalar beta. If beta is zero, C need not be set on input.
                 * @param [in,out] aDataC
                 * input and output matrix C, will be used as an input in case
                 * aBeta != 0
                 * @param [in] aLdc
                 * Leading dimension of matrix C
                 *
                 */
                virtual
                void
                Syrk(const bool &aFillLower, const bool &aTranspose,
                     const int &aNumRowA, const int &aNumColA, const T &aAlpha,
                     const T *aDataA, const int &aLda, const T &aBeta,
                     T *aDataC, const int &aLdc) = 0;

                /**
                 * @brief
                 * Solve the triangular matrix-vector equation.
                 * Solves :  op(A) X = alpha B
                 *           X op(A) = alpha B
                 *
                 *
                 * @param [in] aLeftSide
                 * if True, the operation will be as follow:
                 * op(A) X = alpha B
                 * otherwise, X op(A) = alpha B
                 * @param [in] aFillUpperTri
                 * if true, A is upper triangle, otherwise, lower triangle.
                 * @param [in] aTranspose
                 * if True, op(A) = A^T, otherwise, op(A) = A
                 * @param [in] aNumRowsB
                 * Number of rows in matrix B
                 * @param [in] aNumColsB
                 * Number of cols in matrix B
                 * @param [in] aAlpha
                 * 	Scalar alpha. If alpha is zero, A is not accessed.
                 * @param [in] aDataA
                 * matrix A data
                 * @param [in] aLda
                 * Leading dimension of matrix A
                 * @param [in,out] aDataB
                 * input and output matrix B,On entry.
                 * On exit, overwritten by the solution matrix X.
                 * @param [in] aLdb
                 * Leading dimension of matrix B
                 *
                 */
                virtual
                void
                Trsm(const bool &aLeftSide, const bool &aFillUpperTri,
                     const bool &aTranspose, const int &aNumRowsB,
                     const int &aNumColsB, const T &aAlpha, const T *aDataA,
                     const int &aLda, T *aDataB, const int &aLdb) = 0;

                /**
                 * @brief
                 * Computes the Cholesky factorization of a symmetric
                 * (Hermitian) positive-definite matrix.
                 *
                 * @param [in] aFillUpperTri
                 * Indicates whether the upper or lower triangular part of A is
                 * stored, and how A is factored
                 * @param [in] aNumRow
                 * Order of matrix A, for the whole matrix number of rows or cols
                 * can be used.
                 * @param [in,out] aDataA
                 * Matrix A data, on exit, the buffer will hold the decomposition
                 * output either in the upper or the lower triangle only.
                 * @param [in] aLda
                 * Leading dimension of matrix A
                 *
                 * @returns
                 * rc code : if = 0, the execution is successful.
                 * If rc = -i, the i-th parameter had an illegal value.
                 * If rc = i, the leading minor of order i
                 * (and therefore the matrix A itself) is not positive-definite,
                 * and the factorization could not be completed.
                 *
                 */
                virtual
                int
                Potrf(const bool &aFillUpperTri, const int &aNumRow, T *aDataA,
                      const int &aLda) = 0;

                /**
                 * @brief
                 * Computes the inverse of a symmetric (Hermitian)
                 * positive-definite matrix using the Cholesky factorization.
                 *
                 * @param [in] aFillUpperTri
                 * Indicates whether the upper or lower triangular part of A is
                 * stored, and how A is factored
                 * @param [in] aNumRow
                 * Order of matrix A, for the whole matrix number of rows or cols
                 * can be used.
                 * @param [in,out] aDataA
                 * Matrix A data, on exit, the buffer will hold inv(A).
                 * @param [in] aLda
                 * Leading dimension of matrix A
                 *
                 * @returns
                 * rc code : if = 0, the execution is successful.
                 * If rc = -i, the i-th parameter had an illegal value.
                 * If rc = i, the i-th diagonal element of the Cholesky factor
                 * (and therefore the factor itself) is zero, and the inversion
                 * could not be completed.
                 *
                 */
                virtual
                int
                Potri(const bool &aFillUpperTri, const int &aNumRow, T *aDataA,
                      const int &aLda) = 0;

                /**
                 * @brief
                 * Computes the solution to system of linear equations A * X = B
                 * for GE matrices
                 * The LU decomposition with partial pivoting and row
                 * interchanges is used to factor A as: A = P * L * U,
                 *
                 * where P is a permutation matrix, L is unit lower triangular,
                 * and U is upper triangular.
                 * The factored form of A is then used to solve the system of
                 * equations A * X = B.
                 *
                 * @param [in] aNumN
                 * Number of linear equations, the order of the matrix.
                 * @param [in] aNumNRH
                 * Number of right hand sides, the number of cols of matrix B
                 * @param [in,out] aDataA
                 *  On entry, the N-by-N coefficient matrix A.
                 *  On exit, the factors L and U from the factorization
                 * @param [in] aLda
                 * Leading dimension of matrix A
                 * @param [in,out] aIpiv
                 * The pivot indices that define the permutation matrix P;
                 * row i of the matrix was interchanged with row aIpiv(i).
                 * @param [in,out] aDataOut
                 *  On entry, the N-by-NRHS matrix of right hand side matrix B.
                 *  On exit, if rc = 0, the N-by-NRHS solution matrix X.
                 * @param [in] aLdo
                 * Leading dimension of matrix output
                 *
                 * @returns
                 * rc code :
                 *  = 0:  successful exit
                 *  < 0:  if rc = -i, the i-th argument had an illegal value
                 *  > 0:  if rc = i, U(i,i) is exactly zero.  The factorization
                 * has been completed, but the factor U is exactly singular,
                 * so the solution could not be computed
                 *
                 */
                virtual
                int
                Gesv(const int &aNumN, const int &aNumNRH, T *aDataA,
                     const int &aLda, void *aIpiv, T *aDataOut,
                     const int &aLdo) = 0;

                /**
                 * @brief
                 * Computes an LU factorization of a general M-by-N matrix A
                 * using partial pivoting with row interchanges.
                 *
                 * The factorization has the form
                 *    A = P * L * U
                 * where P is a permutation matrix, L is lower triangular with unit
                 * diagonal elements (lower trapezoidal if m > n), and U is upper
                 * triangular (upper trapezoidal if m < n).
                 *
                 * @param [in] aNumRow
                 * The number of rows of the matrix A
                 * @param [in] aNumCol
                 * The number of columns of the matrix A
                 * @param [in,out] aDataA
                 * On entry, the M-by-N matrix to be factored.
                 * On exit, the factors L and U from the factorization
                 * @param [in] aLda
                 * Leading dimension of matrix A
                 * @param [in,out] aIpiv
                 *  The pivot indices; for 1 <= i <= min(M,N), row i of the
                 *  matrix was interchanged with row aIpiv(i).
                 *
                 * @returns
                 * rc code :
                 *  = 0:  successful exit
                 *  < 0:  if rc = -i, the i-th argument had an illegal value
                 *  > 0:  if rc = i, U(i,i) is exactly zero. The factorization
                 *  has been completed, but the factor U is exactly singular,
                 *  and division by zero will occur if it is used to solve a
                 *  system of equations.
                 *
                 */
                virtual
                int
                Getrf(const int &aNumRow, const int &aNumCol, T *aDataA,
                      const int &aLda, int64_t *aIpiv) = 0;

                /**
                 * @brief
                 * Computes the inverse of a matrix using the LU factorization
                 * computed by GETRF.
                 *
                 * This method inverts U and then computes inv(A) by solving the system
                 * inv(A)*L = inv(U) for inv(A).
                 *
                 * @param [in] aMatRank
                 * The order of the matrix A
                 * @param [in,out] aDataA
                 * On entry, the factors L and U from the factorization
                 * A = P*L*U as computed by GETRF.
                 * On exit, if rc = 0, the inverse of the original matrix A.
                 * @param [in] aLda
                 * Leading dimension of matrix A
                 * @param [in,out] aIpiv
                 * The pivot indices from GETRF; for 1<=i<=N, row i of the
                 * matrix was interchanged with row aIpiv(i)
                 *
                 * @returns
                 * rc code :
                 *  = 0:  successful exit
                 *  < 0:  if rc = -i, the i-th argument had an illegal value
                 *  > 0:  if rc = i, U(i,i) is exactly zero; the matrix is
                 * singular and its inverse could not be computed.
                 *
                 */
                virtual
                int
                Getri(const int &aMatRank, T *aDataA, const int &aLda,
                      int64_t *aIpiv) = 0;


                //svd - lapack  -> gesvd (GPU) and gesdd(CPU)
                virtual
                int
                SVD(const signed char &aJob, const int &aNumRow,
                    const int &aNumCol, T *aDataA, const int &aLda, T *aDataS,
                    T *aDataU, const int &aLdu, T *aDataVT,
                    const int &aLdvt) = 0;

                /**
                 * @brief
                 * Computes the eigenvalues and, optionally, the left and/or
                 * right eigenvectors for SY matrices
                 *
                 * @param [in] aJobzNoVec
                 * If True, Compute eigenvalues only, otherwise, compute both
                 * eigenvalues and eigenvectors.
                 * @param [in] aFillUpperTri
                 * If True, Upper triangle is stored, otherwise, lower triangle.
                 * @param [in] aNumCol
                 * Number of Cols, Matrix order.
                 * @param [in,out] aDataA
                 * On entry, the symmetric matrix A.
                 * On exit, eigenvectors of the matrix A, if aJobzNoVec= False,
                 * otherwise, upper/lower triangle of A, including the diagonal,
                 * is destroyed.
                 * @param[in] aLda
                 * Leading dimension of matrix A
                 * @param[in,out] aDataW
                 * The eigenvalues in ascending order.
                 *
                 * @returns
                 * rc code :
                 *  = 0:  successful exit
                 *  < 0:  if rc = -i, the i-th argument had an illegal value
                 *  > 0:  if rc = i and aJobzNoVec = True, then the algorithm failed
                 * to converge; i off-diagonal elements of an intermediate
                 * tridiagonal form did not converge to zero;
                 * if rc = i and aJobzNoVec = False, then the algorithm failed
                 * to compute an eigenvalue while working on the submatrix
                 * lying in rows and columns rc/(N+1) through
                 * mod(rc,N+1).
                 *
                 */
                virtual
                int
                Syevd(const bool &aJobzNoVec, const bool &aFillUpperTri,
                      const int &aNumCol, T *aDataA, const int64_t &aLda,
                      T *aDataW) = 0;

                /**
                 * @brief
                 * Computes a QR factorization with column pivoting of a
                 *  matrix A:  A*P = Q*R
                 *
                 * @param [in] aNumRow
                 * Number of rows of matrix A
                 * @param [in] aNumCol
                 * Number of cols of matrix A
                 * @param [in,out] aDataA
                 * On entry, the matrix A.
                 * On exit, the upper triangle of the array contains the
                 * min(M,N)-by-N upper trapezoidal matrix R; the elements below
                 * the diagonal, together with the array TAU, represent the
                 * orthogonal matrix Q as a product of min(M,N) elementary
                 * reflectors.
                 * @param [in] aLda
                 * Leading dimension of matrix A
                 * @param[in,out] aJpVt
                 * On entry, if aJpVt(J).ne.0, the J-th column of A is permuted
                 * to the front of A*P (a leading column); if JPVT(J)=0,
                 * the J-th column of A is a free column.
                 * On exit, if JPVT(J)=K, then the J-th column of A*P was the
                 * the K-th column of A.
                 * @param[in,out] aTaw
                 * The scalar factors of the elementary reflectors.
                 *
                 * @returns
                 * rc code :
                 *   = 0: successful exit.
                 *   < 0: if rc = -i, the i-th argument had an illegal value.
                 *
                 */
                virtual
                int
                Geqp3(const int &aNumRow, const int &aNumCol, T *aDataA,
                      const int &aLda, int64_t *aJpVt, T *aTaw) = 0;

                /**
                 * @brief
                 * Generates an NumRow-by-Num real matrix Q with orthonormal columns,
                 * which is defined as the first N columns of a product of K
                 * elementary reflectors of order M
                 *     Q  =  H(1) H(2) . . . H(k)
                 *
                 * @param [in] aNumRow
                 * Number of rows of matrix Q
                 * @param [in] aNum
                 * Number of cols of matrix Q
                 * @param [in] aNumCol
                 * The number of elementary reflectors whose product defines the
                 * matrix Q
                 * @param [in,out] aDataA
                 * On entry, the i-th column must contain the vector which
                 * defines the elementary reflector
                 * On exit, the M-by-N matrix Q.
                 * @param[in] aLda
                 * Leading dimension of matrix Q
                 * @param[in] aTau
                 * Contains the scalar factor of the elementary reflector H(i)
                 *
                 * @returns
                 * rc code :
                 *   = 0: successful exit.
                 *   < 0: if rc = -i, the i-th argument had an illegal value.
                 *
                 */
                virtual
                int
                Orgqr(const int &aNumRow, const int &aNum, const int &aNumCol,
                      T *aDataA, const int &aLda, const T *aTau) = 0;

                /**
                 * @brief
                 * Estimates the reciprocal of the condition number of a general
                 * real matrix A, in either the 1-norm or the infinity-norm, using
                 * the LU factorization.
                 *
                 * @param [in] aNorm
                 * Specifies whether the 1-norm condition number or the
                 * infinity-norm condition number is required:
                 * = 'I' :               Infinity-norm.
                 * = Otherwise :         1-norm.
                 * @param [in] aNumRow
                 *  Number of rows of matrix A, The order of the matrix A.
                 * @param [in] aData
                 * The factors L and U from the factorization A = P*L*U
                 * as computed by GETRF.
                 * @param [in] aLda
                 * Leading dimension of matrix A
                 * @param[in] aNormVal
                 * Leading dimension of matrix Q
                 * @param[out] aRCond
                 * If NORM =  1-norm, the 1-norm of the original matrix A.
                 * If NORM =  Infinity-norm, the infinity-norm of the original matrix A.
                 *
                 * @returns
                 * rc code :
                 *  = 0:  successful exit
                 *  < 0:  if rc = -i, the i-th argument had an illegal value.
                 *        NaNs are illegal values for ANORM, and they propagate to
                 *        the output parameter RCOND.
                 *        Infinity is illegal for ANORM, and it propagates to the output
                 *        parameter RCOND as 0.
                 *  = 1:  if RCOND = NaN, or
                 *           RCOND = Inf, or
                 *           the computed norm of the inverse of A is 0.
                 *        In the latter, RCOND = 0 is returned.
                 *
                 */
                virtual
                int
                Gecon(const std::string &aNorm, const int &aNumRow,
                      const T *aData, const int &aLda, T aNormVal,
                      T *aRCond) = 0;

                /**
                 * @brief
                 * Estimates the reciprocal of the condition number of a
                 * triangular matrix A, in either the 1-norm or the infinity-norm.
                 * The norm of A is computed and an estimate is obtained for
                 * norm(inv(A)), then the reciprocal of the condition number is
                 * computed as
                 *    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
                 *
                 * @param [in] aNorm
                 * Specifies whether the 1-norm condition number or the
                 * infinity-norm condition number is required:
                 * = 'I' :               Infinity-norm.
                 * = Otherwise :         1-norm.
                 * @param [in] aUpperTriangle
                 *  If True, A is upper triangle, otherwise lower triangle.
                 * @param [in] aUnitTriangle
                 * If True, A us unit triangular, otherwise A is non-unit triangular
                 * @param [in] aMatOrder
                 * The order of matrix A.
                 * @param[in] aData
                 * The triangular matrix A.  If upper triangle, the leading N-by-N
                 * upper triangular part of the array A contains the upper
                 * triangular matrix, and the strictly lower triangular part of
                 * A is not referenced.  If lower triangle, the leading N-by-N lower
                 * triangular part of the array A contains the lower triangular
                 * matrix, and the strictly upper triangular part of A is not
                 * referenced.  If unit triangular, the diagonal elements of A are
                 * also not referenced and are assumed to be 1.
                 * @param[in] aLda
                 * Leading dimension of matrix A.
                 * @param[out] aRCond
                 *  The reciprocal of the condition number of the matrix A,
                 *  computed as RCOND = 1/(norm(A) * norm(inv(A))).
                 *
                 * @returns
                 * rc code :
                 *  = 0:  successful exit
                 *  < 0:  if rc = -i, the i-th argument had an illegal value
                 */
                virtual
                int
                Trcon(const std::string &aNorm, const bool &aUpperTriangle,
                      const bool &aUnitTriangle, const int &aMatOrder,
                      const T *aData, const int &aLda, T *aRCond) = 0;


            };

            /** Macro to instantiate the class as float and double **/
            MPCR_INSTANTIATE_CLASS(LinearAlgebraBackend)
        }
    }
}


#endif //MPCR_LINEARALGEBRABACKEND_HPP
