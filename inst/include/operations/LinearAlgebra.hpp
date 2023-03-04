
#ifndef MPR_LINEARALGEBRA_HPP
#define MPR_LINEARALGEBRA_HPP


#include <data-units/DataType.hpp>


#define LAYOUT blas::Layout::ColMajor


namespace mpr {
    namespace operations {
        namespace linear {

            /**
             * @brief
             * Calculate CrossProduct of 2 MPR Matrices
             * performs:
             * x %*% t(x) , x %*% t(y) , x %*% y  , t(x) %*% y
             *
             * @param[in] aInputA
             * MPR Matrix
             * @param[in] aInputB
             * MPR Matrix
             * @param[out] aOutput
             * MPR Matrix
             * @param[in] aTransposeA
             * bool to indicate whether aInputA should be Transposed or not
             * @param[in] aTransposeB
             * bool to indicate whether aInputB should be Transposed or not
             *
             */
            template <typename T>
            void
            CrossProduct(DataType &aInputA, DataType &aInputB,
                         DataType &aOutput, const bool &aTransposeA,
                         const bool &aTransposeB);

            /**
             * @brief
             * Check if a Matrix Is Symmetric
             *
             * @param[in] aInput
             * MPR Matrix
             * @param[out] aOutput
             * true if symmetric ,false otherwise
             *
             */
            template <typename T>
            void
            IsSymmetric(DataType &aInput, bool &aOutput);

            /**
             * @brief
             * Calculate Cholesky decomposition
             *
             * @param[in] aInput
             * MPR Matrix
             * @param[out] aOutput
             * MPR Matrix containing decomposition result
             *
             */
            template <typename T>
            void
            Cholesky(DataType &aInputA, DataType &aOutput);

            /**
             * @brief
             * Invert a symmetric, positive definite square matrix from its
             * Cholesky decomposition.
             *
             * @param[in] aInput
             * MPR Matrix containing Cholesky decomposition.
             * @param[out] aOutput
             * MPR Matrix
             *
             */
            template <typename T>
            void
            CholeskyInv(DataType &aInputA, DataType &aOutput,
                        const size_t &aNCol);

            /**
             * @brief
             * Solves the equation AX=B
             *
             * @param[in] aInputA
             * MPR Matrix A
             * @param[in] aInputB
             * MPR Matrix X
             * @param[out] aOutput
             * MPR Matrix B
             * @param[in] aSingle
             * if true only aInputA will be used and for X t(A) will be used.
             *
             */
            template <typename T>
            void
            Solve(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                  const bool &aSingle);


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
             * @param[out] aOutput
             * The solution of the triangular system
             * @param[in] aCol
             * The number of columns of aInputA and rows of aInputB to use.
             * @param[in] aUpperTri
             * logical; if true (default), the upper triangular part of aInputA
             * is used. Otherwise, the lower one.
             * @param[in] aTranspose
             * logical; if true, solve  for t(aInputA) %*% aOutput == aInputB.
             */

            template <typename T>
            void
            BackSolve(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                      const size_t &aCol, const bool &aUpperTri,
                      const bool &aTranspose);

            /**
             * @brief
             * Calculate Eigen Values and (optionally) Eigen Vectors.
             * if(apOutputVectors)= nullptr , only the values will be calculated
             *
             * @param[in] aInput
             * MPR Square Matrix
             * @param[out] aOutputValues
             * Eigen Values
             * @param[out] apOutputVectors
             * Eigen Vectors
             */
            template <typename T>
            void
            Eigen(DataType &aInput, DataType &aOutputValues,
                  DataType *apOutputVectors = nullptr);

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
             * @param[out] aOutput
             * Norm Value
             */
            template <typename T>
            void
            Norm(DataType &aInput, const std::string &aType, DataType &aOutput);


            /**
             * @brief
             * Computes the QR decomposition of a matrix.
             *
             * @param[in] aInputA
             * MPR Matrix
             * @param[out] aOutputQr
             * a MPR Matrix with the same dimensions as aInputA. The upper
             * triangle contains the decomposition and the lower triangle
             * contains information of the decomposition (stored in compact form)
             * @param[out] aOutputQraux
             * a vector of length ncol(aInputA) which contains additional
             * information on Q
             * @param[out] aOutputPivot
             * information on the pivoting strategy used during the decomposition
             * @param[out] aRank
             * the rank of aInputA ,always full rank in the LAPACK.
             * @param[in] aTolerance
             * the tolerance for detecting linear dependencies in the columns of
             * aInputA
             *
             */
            template <typename T>
            void
            QRDecomposition(DataType &aInputA, DataType &aOutputQr,
                            DataType &aOutputQraux, DataType &aOutputPivot,
                            DataType &aRank, const double &aTolerance = 1e-07);

            /**
             * @brief
             * returns R. This may be pivoted,
             * The number of rows of R is either nrow(aInputA) or ncol(aInputA)
             * (and may depend on whether complete is TRUE or FALSE).
             *
             * @param[in] aInputA
             * MPR Matrix
             * @param[out] aOutput
             * returns R. This may be pivoted. As MPR Object.
             * @param[in] aComplete
             * logical expression . Indicates whether an arbitrary
             * orthogonal completion of the Q or X matrices is to be made,
             * or whether the  matrix is to be completed by binding zero-value
             * rows beneath the square upper triangle.
             *
             */
            template <typename T>
            void
            QRDecompositionR(DataType &aInputA, DataType &aOutput,
                             const bool &aComplete);

            /**
             * @brief
             * Returns part or all of Q, the order-nrow(X) orthogonal (unitary)
             * transformation represented by qr. If complete is TRUE, Q has
             * nrow(X) columns. If complete is FALSE, Q has ncol(X) columns.
             *
             * @param[in] aInputA
             * MPR Matrix QR
             * @param[in] aInputB
             * MPR Object Representing QRAUX
             * @param[out] aOutput
             * returns Q. As MPR Object.
             * @param[in] aComplete
             * logical expression . Indicates whether an arbitrary
             * orthogonal completion of the Q or X matrices is to be made,
             * or whether the  matrix is to be completed by binding zero-value
             * rows beneath the square upper triangle.
             *
             */
            template <typename T>
            void
            QRDecompositionQ(DataType &aInputA, DataType &aInputB,
                             DataType &aOutput, const bool &aComplete);

            /**
             * @brief
             * Returns part or all of Q, the order-nrow(X) orthogonal (unitary)
             * transformation represented by qr. If complete is TRUE, Q has
             * nrow(X) columns. If complete is FALSE, Q has ncol(X) columns.
             * and each column of Q is multiplied by the corresponding value in Dvec.
             * @param[in] aInputA
             * MPR Matrix
             * @param[in] aInputB
             * MPR Object Representing QRAUX
             * @param[in] aInputC
             * MPR Object Representing DVec
             * @param[out] aOutput
             * returns Q. As MPR Object.
             * @param[in] aComplete
             * logical expression . Indicates whether an arbitrary
             * orthogonal completion of the Q or X matrices is to be made,
             * or whether the  matrix is to be completed by binding zero-value
             * rows beneath the square upper triangle.
             *
             */
            template <typename T>
            void
            QRDecompositionQY(DataType &aInputA, DataType &aInputB,
                              DataType &aInputC, DataType &aOutput,
                              const bool &aTranspose);

            /**
             * @brief
             * Compute the singular-value decomposition of a rectangular matrix.
             *
             * @param[in] aInputA
             * MPR Matrix
             * @param[out] aOutputS
             * an MPR Object containing the singular values of aInputA,
             * of length min(n, p).
             * @param[out] aOutputU
             * a MPR Matrix whose columns contain the left singular vectors of
             * aInputA, present if nu > 0. Dimension c(m, nu).
             * @param[out] aOutputV
             * a MPR Matrix whose columns contain the right singular vectors of
             * aInputA, present if nv > 0. Dimension c(n, nv).
             * @param[in] aNu
             * the number of left singular vectors to be computed.
             * This must between 0 and m = nrow(aInputA).
             * @param[in] aNv
             * the number of right singular vectors to be computed.
             * This must be between 0 and n = ncol(aInputA).
             * @param[in] aTranspose
             * Bool if true, aOutputV will contain V ,otherwise VT
             *
             */
            template <typename T>
            void
            SVD(DataType &aInputA, DataType &aOutputS, DataType &aOutputU,
                DataType &aOutputV, const size_t &aNu,
                const size_t &aNv, const bool &aTranspose = true);

            /**
             * @brief
             * Estimate the reciprocal of the condition number of a matrix.
             *
             * @param[in] aInput
             * MPR Matrix
             * @param[out] aOutput
             * MPR Vector containing one element which is an estimate of the
             * reciprocal condition number of aInput.
             * @param[in] aNorm
             * character string indicating the type of norm to be used in the
             * estimate. The default is "O" for the 1-norm ("O" is equivalent to "1").
             * For sparse matrices,  when useInv=TRUE, norm can be any of the kinds allowed for norm;
             * otherwise, the other possible value is "I" for the infinity norm.
             * @param[in] aTriangle
             * Bool if true,Only the lower triangle will be used.
             *
             */
            template <typename T>
            void
            ReciprocalCondition(DataType &aInput, DataType &aOutput,
                                const std::string &aNorm,
                                const bool &aTriangle);
        }
    }
}

#endif //MPR_LINEARALGEBRA_HPP
