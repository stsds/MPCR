

#ifndef MPCR_HELPERS_HPP
#define MPCR_HELPERS_HPP


#include <utilities/MPCRDispatcher.hpp>
#include <data-units/DataType.hpp>


namespace mpcr {
    namespace operations {
        namespace helpers {
            template <typename T>
            class Helpers {

            public:

                /**
                 * @brief
                 * Default constructor
                 *
                 */
                Helpers() = default;

                /**
                 * @brief
                 * Default de-constructor
                 *
                 */
                ~Helpers() = default;

                /**
                 * @brief
                 * Symmetries a matrix by copying the lower to the upper triangle
                 * and vice verse.
                 *
                 * @param [in,out] aInput
                 * MPCR object to symmetries.
                 * @param [in] aToUpperTriangle
                 * Flag to decide which triangle needs to be filled.
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                Symmetrize(DataType &aInput, const bool &aToUpperTriangle,
                           kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Reverse a matrix or vector
                 *
                 * @param [in,out] aInput
                 * MPCR object to Reverse
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                Reverse(DataType &aInput, kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Transpose a matrix
                 *
                 * @param [in,out] aInput
                 * MPCR object to transpose
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                Transpose(DataType &aInput, kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Fill upper or lower triangle with a specific value
                 *
                 * @param [in,out] aInput
                 * MPCR object to fill
                 * @param [in] aValue
                 * Value used to fill the triangle with.
                 * @param [in] aUpperTriangle
                 * Flag to indicate which triangle to fill.
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                FillTriangle(DataType &aInput, const double &aValue,
                             const bool &aUpperTriangle,
                             kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Create an identity matrix
                 *
                 * @param [out] apData
                 * Data buffer allocated used as an output
                 * @param [in] aSideLength
                 * Matrix side length.
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                CreateIdentityMatrix(T *apData, size_t &aSideLength,
                                     kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Calculate Norm using the maximum absolute row sum.
                 *
                 * @param [in] aInput
                 * MPCR Matrix
                 * @param [out] aValue
                 * Norm value
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                NormMARS(DataType &aInput, T &aValue,
                         kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Calculate Norm using the maximum absolute column sum.
                 *
                 * @param [in] aInput
                 * MPCR Matrix
                 * @param [out] aValue
                 * Norm value
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                NormMACS(DataType &aInput, T &aValue,
                         kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Calculate the Euclidean norm.
                 *
                 * @param [in] aInput
                 * MPCR Matrix
                 * @param [out] aValue
                 * Norm value
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                NormEuclidean(DataType &aInput, T &aValue,
                              kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Calculate Norm using the maximum modulus of all the elements.
                 *
                 * @param [in] aInput
                 * MPCR Matrix
                 * @param [out] aValue
                 * Norm value
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                NormMaxMod(DataType &aInput, T &aValue,
                           kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Calculate matrix rank.
                 *
                 * @param [in] aInput
                 * MPCR Matrix.
                 * @param [out] aRank
                 * Matrix Rank.
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                GetRank(DataType &aInput, T &aRank,
                        kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Check whether the matrix is symmetric or not.
                 *
                 * @param [in] aInput
                 * MPCR Matrix.
                 * @param [out] aOutput
                 * flag indicating whether the matrix is symmetric or not.
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                IsSymmetric(DataType &aInput, bool &aOutput,
                            kernels::RunContext *aContext) = 0;

                /**
                 * @brief
                 * Copy upper triangle for one MPCR object to another.
                 * Used in QR-R Kernel.
                 *
                 * @param [in] aInput
                 * MPCR Matrix.
                 * @param [out] aOutput
                 * MPCR output matrix.
                 * @param [in] aContext
                 * Run context used for GPU helpers, can be null in case of CPU.
                 *
                 */
                virtual
                void
                CopyUpperTriangle(DataType &aInput, DataType &aOutput,
                                  kernels::RunContext *aContext) = 0;

            };

            MPCR_INSTANTIATE_CLASS(Helpers)
        }
    }
}
#endif //MPCR_HELPERS_HPP
