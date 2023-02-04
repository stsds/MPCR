
#ifndef MPR_MATHEMATICALOPERATIONS_HPP
#define MPR_MATHEMATICALOPERATIONS_HPP

#include <data-units/DataType.hpp>


namespace mpr {
    namespace operations {
        namespace math {

            template <typename T>
            void
            PerformRoundOperation(DataType &aInputA, DataType &aOutput,
                                  std::string aFun);


            template <typename T>
            void
            SquareRoot(DataType &aInputA, DataType &aOutput);


            template <typename T>
            void
            Exponential(DataType &aInputA, DataType &aOutput,
                        bool aFlag = false);


            template <typename T>
            void
            IsFinite(DataType &aInputA, std::vector <bool> &aOutput);

            template <typename T>
            void
            IsInFinite(DataType &aInputA, std::vector <int> &aOutput);

            template <typename T>
            void
            Log(DataType &aInputA, DataType &aOutput, int aBase = 10);

            template <typename T>
            void
            PerformTrigOperation(DataType &aInputA, DataType &aOutput,
                                 std::string aFun);

            template <typename T>
            void
            PerformInverseTrigOperation(DataType &aInputA, DataType &aOutput,
                                        std::string aFun);


        }
    }
}


#endif //MPR_MATHEMATICALOPERATIONS_HPP
