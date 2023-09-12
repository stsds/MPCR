/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_TILELINEARALGEBRA_HPP
#define MPCR_TILELINEARALGEBRA_HPP

#include <data-units/MPCRTile.hpp>


namespace mpcr {
    namespace operations {
        namespace linear {

            /**
             * @brief
             * Calculate Cholesky decomposition for Tiled-Symmetric Matrix
             *
             * @param[in] aMatrix
             * MPCR Matrix
             * @param[in] aOverWriteInput
             * if true , the input will be overwritten with the output,otherwise
             * a new copy will be created.
             * @param[in] aNumThreads
             * int to decide the number of threads used in OpenMP, default = 1
             * (sequential)
             * @returns
             * MPCRTile Matrix containing decomposition result.
             *
             */
            MPCRTile *
            TileCholesky(MPCRTile &aMatrix, const bool &aOverWriteInput = true,
                         const unsigned int &aNumThreads = 1);

            /**
             * @brief
             * Tiled-Matrix Multiplication of 2 MPCR Tile Matrices
             * performs:
             * C = alpha A * B + beta C
             *
             * @param[in] aInputA
             * MPCRTile Matrix
             * @param[in] aInputB
             * MPCRTile Matrix
             * @param[in,out] aInputC
             * MPCRTile Matrix
             * @param[in] aTransposeA
             * flag to indicate whether A should be transposed
             * @param[in] aTransposeB
             * flag to indicate whether B should be transposed
             * @param[in] aAlpha
             * Factor multiplied to Matrix A
             * @param [in] aBeta
             * Factor multiplied to Matrix C
             * @param[in] aNumThreads
             * int to decide the number of threads used in OpenMP, default = 1
             * (sequential)
             *
             */
            void
            TileGemm(MPCRTile &aInputA, MPCRTile &aInputB, MPCRTile &aInputC,
                     const bool &aTransposeA = false,
                     const bool &aTransposeB = false,
                     const double &aAlpha = 1, const double &aBeta = 1,
                     const unsigned int &aNumThreads = 1);


            /**
             * @brief
             * Solves a triangular matrix equation
             * performs:
             * op(A)*X=alpha*B
             * X*op(A)=alpha*B
             *
             * @param[in] aInputA
             * MPCRTile Matrix A
             * @param[in,out] aInputB
             * MPCRTile Matrix B, X after returning.
             * @param[in] aSide
             * 'R for Right side , 'L' for Left side
             * @param [in] aUpperTriangle
             * What part of the matrix A is referenced, the opposite triangle
             * being assumed to be zero.
             * if true ,the Upper triangle is referenced. otherwise the Lower.
             * @param[in] aTranspose
             * if true , the transpose of A is used.
             * @param[in] aAlpha
             * Factor used for A , If alpha is zero, A is not accessed.
             *
             *
             */
            void
            TileTrsm(MPCRTile &aInputA, MPCRTile &aInputB, const char &aSide,
                     const bool &aUpperTriangle, const bool &aTranspose,
                     const double &aAlpha);
        }
    }
}


#endif //MPCR_TILELINEARALGEBRA_HPP
