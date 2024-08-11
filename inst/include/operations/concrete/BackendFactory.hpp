

#ifndef MPCR_BACKENDFACTORY_HPP
#define MPCR_BACKENDFACTORY_HPP


#include <common/Definitions.hpp>
#include <utilities/MPCRErrorHandler.hpp>
#include <operations/concrete/CPULinearAlgebra.hpp>
#include <operations/concrete/CPUHelpers.hpp>


#ifdef USE_CUDA
#include <operations/concrete/GPULinearAlgerba.hpp>
#include <operations/concrete/GPUHelpers.hpp>
#endif


namespace mpcr::operations {
        /**
         * Template Linear Algebra backend Factory
         * This Factory is intended to be used on Float and Double only.
         **/
        template <typename T>
        class BackendFactory {
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
            std::unique_ptr<linear::LinearAlgebraBackend <T>>
            CreateLinearAlgebraBackend(const definitions::OperationPlacement &aPlacement) {
                if (aPlacement == definitions::CPU) {
                    return std::make_unique <linear::CPULinearAlgebra <T>>();
                } else {
#ifdef USE_CUDA
                    return  std::make_unique <linear::GPULinearAlgebra <T>>();
#else
                    MPCR_API_EXCEPTION("Package is built without GPU support, Cannot retrieve GPU Linear algebra backend",-1);
                    return nullptr;
#endif
                }
            }


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
            std::unique_ptr<helpers::Helpers <T>>
            CreateHelpersBackend(const definitions::OperationPlacement &aPlacement) {
                if (aPlacement == definitions::CPU) {
                    return std::make_unique <helpers::CPUHelpers<T>>();
                } else {
#ifdef USE_CUDA
                    return  std::make_unique <helpers::GPUHelpers<T>>();
#else
                    MPCR_API_EXCEPTION("Package is built without GPU support, Cannot retrieve GPU Linear algebra backend",-1);
                    return nullptr;
#endif
                }
            }
        };

        MPCR_INSTANTIATE_CLASS(BackendFactory)

}
#endif //MPCR_BACKENDFACTORY_HPP
