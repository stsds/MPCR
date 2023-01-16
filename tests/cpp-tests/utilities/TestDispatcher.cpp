
#include <libraries/catch/catch.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <data-units/DataType.hpp>


using namespace mpr::precision;


template<typename T>
void
TestSimpleDispatch(T aNumA, T aNumB, bool &aIsEqual) {
    if (aNumA == aNumB) {
        aIsEqual = true;
    } else {
        aIsEqual = false;
    }
}


template<typename T, typename X, typename Y>
void
TestComplexDispatch(DataType *apNumA, DataType *apNumB, DataType *apNumC) {
    T *data_one = (T *) apNumA->GetData();
    X *data_two = (X *) apNumB->GetData();
    Y *data_out = (Y *) apNumC->GetData();

    REQUIRE(apNumA->GetSize() == apNumB->GetSize());

    for (auto i = 0; i < apNumA->GetSize(); i++) {
        REQUIRE(data_out[i] == data_one[i] + data_two[i]);
    }
}


template<typename T>
void
GenerateData(DataType *apDataType, double aVal) {
    T *data = (T *) apDataType->GetData();
    auto size = apDataType->GetSize();
    for (auto i = 0; i < size; i++) {
        data[i] = aVal;
    }
}


void
TEST_DISPATCHER() {

    bool rc = false;
    int a = 5;
    int b = 10;
    Precision precision = INT;
    SIMPLE_DISPATCH(precision, TestSimpleDispatch, a, b, rc)
    REQUIRE(rc == false);

    a = 10;
    SIMPLE_DISPATCH(precision, TestSimpleDispatch, a, b, rc)
    REQUIRE(rc == true);
    rc = false;

    float temp_float_a = 10.332;
    SIMPLE_DISPATCH(precision, TestSimpleDispatch, temp_float_a, b, rc)
    REQUIRE(rc == true);

    float temp_float_b = 10.332;
    rc = false;
    precision = FLOAT;
    SIMPLE_DISPATCH(precision, TestSimpleDispatch, temp_float_a, temp_float_b,
                    rc)
    REQUIRE(rc == true);

    DataType dataA(50, FLOAT);
    DataType dataB(50, FLOAT);
    DataType dataOut(50, FLOAT);

    SIMPLE_DISPATCH(FLOAT, GenerateData, &dataOut, 3)

    precision = FFF;
    DISPATCHER(precision, TestComplexDispatch, &dataA, &dataB, &dataOut)


}


TEST_CASE("Dispatcher Test", "[Dispatcher]") {
    TEST_DISPATCHER();
}

