

#include <operations/MathematicalOperations.hpp>
#include <operations/helpers/MathematicalOperationsHelper.hpp>


using namespace mpr::operations;


template <typename T>
void math::PerformRoundOperation(DataType &aInputA, DataType &aOutput,
                                 std::string aFun) {

    auto pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto pOutput = new T[size];

    PERFORM_ROUND_OP(pData, pOutput, size, aFun)

    aOutput.ClearUp();
    aOutput.SetDimensions(aInputA);
    aOutput.SetData((char *) pOutput);

}


template <typename T>
void math::SquareRoot(DataType &aInputA, DataType &aOutput) {

    auto pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto pOutput = new T[size];

    try {
        for (auto i = 0; i < size; i++) {
            pOutput[ i ] = std::sqrt(pData[ i ]);
        }
    } catch (...) {
        MPR_API_EXCEPTION("Cannot Perform SQRT on Negative Values", -1);
    }


    aOutput.ClearUp();
    aOutput.SetDimensions(aInputA);
    aOutput.SetData((char *) pOutput);
}


template <typename T>
void math::Exponential(DataType &aInputA, DataType &aOutput, bool aFlag) {

    auto pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto pOutput = new T[size];
    double val = 0.0;
    if (aFlag) {
        val = 1.0;
    }

    for (auto i = 0; i < size; i++) {
        pOutput[ i ] = std::exp(pData[ i ]) - val;
    }

    aOutput.ClearUp();
    aOutput.SetDimensions(aInputA);
    aOutput.SetData((char *) pOutput);
}


template <typename T>
void math::IsFinite(DataType &aInputA, std::vector <int> &aOutput) {

    auto pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    aOutput.clear();
    aOutput.resize(size);

    for (auto i = 0; i < size; i++) {
        aOutput[ i ] = std::isfinite(pData[ i ]);
    }
}


template <typename T>
void math::IsInFinite(DataType &aInputA, std::vector <int> &aOutput) {

    auto pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    aOutput.clear();
    aOutput.resize(size);

    for (auto i = 0; i < size; i++) {
        if (!std::isnan(pData[ i ])) {
            aOutput[ i ] = std::isinf(pData[ i ]);
        } else {
            aOutput[ i ] = INT_MIN;
        }
    }
}


template <typename T>
void math::Log(DataType &aInputA, DataType &aOutput, double aBase) {

    auto pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto pOutput = new T[size];

    if (aBase == 10) {
        for (auto i = 0; i < size; i++) {
            pOutput[ i ] = std::log10(pData[ i ]);
        }
    } else if (aBase == 2) {
        for (auto i = 0; i < size; i++) {
            pOutput[ i ] = std::log2(pData[ i ]);
        }
    }else if(aBase==1){

        auto val= 1.0/ log(std::exp(1));

        for (auto i = 0; i < size; i++) {
            pOutput[ i ] = std::log(pData[ i])*val;
        }
    }else {
        delete[] pOutput;
        MPR_API_EXCEPTION("Unknown Log Base", aBase);
    }


    aOutput.ClearUp();
    aOutput.SetDimensions(aInputA);
    aOutput.SetData((char *) pOutput);

}


template <typename T>
void
math::PerformTrigOperation(DataType &aInputA, DataType &aOutput,
                           std::string aFun) {
    auto pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto pOutput = new T[size];

    PERFORM_TRIG_OP(pData, pOutput, size, aFun)

    aOutput.ClearUp();
    aOutput.SetDimensions(aInputA);
    aOutput.SetData((char *) pOutput);

}


template <typename T>
void
math::PerformInverseTrigOperation(DataType &aInputA, DataType &aOutput,
                                  std::string aFun) {
    auto pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto pOutput = new T[size];

    PERFORM_INV_TRIG_OP(pData, pOutput, size, aFun)

    aOutput.ClearUp();
    aOutput.SetDimensions(aInputA);
    aOutput.SetData((char *) pOutput);

}


template <typename T>
void
math::Round(DataType &aInputA, DataType &aOutput, const int &aDecimalPoint) {

    auto pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto pOutput = new T[size];
    auto mult_val = std::pow(10, aDecimalPoint);

    for (auto i = 0; i < size; i++) {
        auto val_temp = pData[ i ] * mult_val;
        val_temp = std::round(val_temp);
        pOutput[ i ] = val_temp / mult_val;
    }

    aOutput.ClearUp();
    aOutput.SetDimensions(aInputA);
    aOutput.SetData((char *) pOutput);
}


template <typename T>
void math::Gamma(DataType &aInputA, DataType &aOutput, const bool &aLGamma) {

    auto pData = (T *) aInputA.GetData();
    auto size = aInputA.GetSize();
    auto pOutput = new T[size];
    if (aLGamma) {
        for (auto i = 0; i < size; i++) {
            pOutput[i] = std::lgamma(pData[ i ]);
        }
    } else {
        for (auto i = 0; i < size; i++) {
            pOutput[i] = std::tgamma(pData[ i ]);
        }
    }

    aOutput.ClearUp();
    aOutput.SetDimensions(aInputA);
    aOutput.SetData((char *) pOutput);
}


SIMPLE_INSTANTIATE(void, math::Exponential, DataType &aInputA,
                   DataType &aOutput,
                   bool aFlag)

SIMPLE_INSTANTIATE(void, math::IsFinite, DataType &aInputA,
                   std::vector <int> &aOutput)

SIMPLE_INSTANTIATE(void, math::IsInFinite, DataType &aInputA,
                   std::vector <int> &aOutput)

SIMPLE_INSTANTIATE(void, math::Log, DataType &aInputA, DataType &aOutput,
                   double aBase)

SIMPLE_INSTANTIATE(void, math::PerformInverseTrigOperation, DataType &aInputA,
                   DataType &aOutput, std::string aFun)

SIMPLE_INSTANTIATE(void, math::PerformRoundOperation, DataType &aInputA,
                   DataType &aOutput, std::string aFun)

SIMPLE_INSTANTIATE(void, math::PerformTrigOperation, DataType &aInputA,
                   DataType &aOutput, std::string aFun)

SIMPLE_INSTANTIATE(void, math::SquareRoot, DataType &aInputA, DataType &aOutput)

SIMPLE_INSTANTIATE(void, math::Round, DataType &aInputA, DataType &aOutput,
                   const int &aDecimalPoint)

SIMPLE_INSTANTIATE(void, math::Gamma, DataType &aInputA, DataType &aOutput,
                   const bool &aLGamma)
