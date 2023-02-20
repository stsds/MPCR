
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
                            size_t &aRank, const double &aTolerance = 1e-07);


            template <typename T>
            void
            QRDecompositionR(DataType &aInputA, DataType &aOutput,
                             const bool &aComplete);

            template <typename T>
            void
            QRDecompositionQ(DataType &aInputA, DataType &aInputB,
                             DataType &aOutput, const bool &aComplete);

            template <typename T>
            void
            QRDecompositionQY(DataType &aInputA, DataType &aInputB,
                              DataType &aInputC, DataType &aOutput,
                              const bool &aTranspose);

            template <typename T>
            void
            SVD(DataType &aInputA, DataType &aOutputS, DataType &aOutputU,
                DataType &aOutputV, const size_t &aNu,
                const size_t &aNv, const bool &aTranspose = true);

            template <typename T>
            void
            ReciprocalCondition(DataType &aInput, DataType &aOutput,
                                const std::string &aNorm,
                                const bool &aTriangle);
        }
    }
}

#endif //MPR_LINEARALGEBRA_HPP
