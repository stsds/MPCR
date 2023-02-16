
#ifndef MPR_LINEARALGEBRA_HPP
#define MPR_LINEARALGEBRA_HPP


#include <data-units/DataType.hpp>


#define LAYOUT blas::Layout::ColMajor


namespace mpr {
    namespace operations {
        namespace linear {

            template <typename T>
            void
            CrossProduct(DataType &aInputA, DataType &aInputB,
                         DataType &aOutput, const bool &aTransposeA,
                         const bool &aTransposeB);

            template <typename T>
            void
            IsSymmetric(DataType &aInput, bool &aOutput);


            template <typename T>
            void
            Cholesky(DataType &aInputA, DataType &aOutput);

            template <typename T>
            void
            CholeskyInv(DataType &aInputA, DataType &aOutput,
                        const size_t &aNCol);


            template <typename T>
            void
            Solve(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                  const bool &aSingle);


            template <typename T>
            void
            BackSolve(DataType &aInputA, DataType &aInputB, DataType &aOutput,
                      const size_t &aCol, const bool &aUpperTri,
                      const bool &aTranspose);

            template <typename T>
            void
            Eigen(DataType &aInput, DataType &aOutputValues,
                  DataType *apOutputVectors = nullptr);


            template <typename T>
            void
            Norm(DataType &aInput, const std::string &aType, DataType &aOutput);


            template <typename T>
            void
            QRDecomposition(DataType &aInputA, DataType &aOutputQr,DataType &aOutputQraux,DataType &aOutputPivot);


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
            SVD(DataType &aInputA, DataType &aOutputS,DataType &aOutputU,DataType &aOutputV, const size_t &aNu,
                const size_t &aNv);

            template <typename T>
            void
            ReciprocalCondition(DataType &aInput, DataType &aOutput,
                                const std::string &aNorm,
                                const bool &aTriangle);
        }
    }
}

#endif //MPR_LINEARALGEBRA_HPP
