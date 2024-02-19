

#ifndef MPCR_LINEARALGEBRABACKENDFACTORY_HPP
#define MPCR_LINEARALGEBRABACKENDFACTORY_HPP


#include <common/Definitions.hpp>
#include <utilities/MPCRErrorHandler.hpp>
#include <operations/concrete/CPULinearAlgebra.hpp>

#ifdef USE_CUDA
#include <operations/concrete/GPULinearAlgerba.hpp>
#endif


namespace mpcr::operations {
    namespace linear {
        /**
         * Template Linear Algebra backend Factory
         * This Factory is intended to be used on Float and Double only.
         **/
        template <typename T>
        class LinearAlgebraBackendFactory {
        public:

            /**
             * @brief
             * Creates a new instance of Linear algebra backend ( CPU/GPU )
             *
             * @param [in] aPlacement
             * if CPU, CPU Linear algebra backend will be created.
             * if GPU, GPU Linear algebra backend will be created.
             * The function will throw error in case GPU is used when the code is
             * not built with GPU support.
             *
             * @returns
             * Unique pointer holder the linear algebra backend.
             */
            static
            std::unique_ptr<LinearAlgebraBackend <T>>
            CreateBackend(const definitions::OperationPlacement &aPlacement) {
                if (aPlacement == definitions::CPU) {
                    return std::make_unique <CPULinearAlgebra <T>>();
                } else {
#ifdef USE_CUDA
                    return  std::make_unique <GPULinearAlgebra <T>>();
#else
                    MPCR_API_EXCEPTION("Package is built without GPU support, Cannot retrieve GPU Linear algebra backend",-1);
                    return nullptr;
#endif
                }
            }

        };

        MPCR_INSTANTIATE_CLASS(LinearAlgebraBackendFactory)
    }
}
#endif //MPCR_LINEARALGEBRABACKENDFACTORY_HPP
