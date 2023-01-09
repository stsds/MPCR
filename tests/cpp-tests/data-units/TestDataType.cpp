
#include <libraries/catch/catch.hpp>
#include <data-units/DataType.hpp>
#include <utilities/MPRDispatcher.hpp>


using namespace std;
using namespace mpr::precision;


template<typename T>
void
CheckValues(DataType *data, char *validator) {

    T *temp_data = (T *) data->GetData();
    T *temp_validate = (T *) validator;
    auto size = data->GetSize();
    for (auto i = 0; i < size; i++) {
        REQUIRE(temp_data[i] == temp_validate[i]);
    }
}


template<typename T>
void
InitValidator(char *&data, size_t size) {
    T *temp = new T[size];
    for (auto i = 0; i < size; i++) {
        temp[i] = (T) 1.5;
    }
    data = (char *) temp;
}


void
TEST_DATA_TYPE() {

    SECTION("Test Initialization") {
        cout << "Testing MPR CLASS ..." << endl;

        DataType a(50, "float");
        REQUIRE(a.GetSize() == 50);
        REQUIRE(a.IsMatrix() == false);
        REQUIRE(a.GetDimensions() == nullptr);

        char *validator;
        Precision temp_precision = FLOAT;
        SIMPLE_DISPATCH(temp_precision, InitValidator, validator, 50)
        SIMPLE_DISPATCH(temp_precision, CheckValues, &a, validator)

        a.ToMatrix(5, 10);
        REQUIRE(a.GetSize() == 50);
        REQUIRE(a.IsMatrix() == true);
        REQUIRE(a.GetDimensions()->GetNRow() == 5);
        REQUIRE(a.GetDimensions()->GetNCol() == 10);

        a.ToVector();
        REQUIRE(a.IsMatrix() == false);
        REQUIRE(a.GetSize() == 50);
        REQUIRE(a.GetDimensions() == nullptr);

        delete[] validator;
    }SECTION("Test Setter and Getter") {
        DataType a(50, "double");
        char *validator;
        Precision temp_precision = DOUBLE;
        SIMPLE_DISPATCH(temp_precision, InitValidator, validator, 50)
        auto data = (double *) validator;
        auto size = a.GetSize();
        for (auto i = 0; i < size; i++) {
            REQUIRE(data[i] == a.GetVal(i));
        }

        for (auto i = 0; i < size; i++) {
            a.SetVal(i, 3.555555555);
        }

        for (auto i = 0; i < size; i++) {
            REQUIRE(a.GetVal(i) == 3.555555555);
        }

        DataType b(50, "float");

        SECTION("Test Copy Constructor") {
            DataType c = b;

            REQUIRE(c.GetSize() == 50);
            REQUIRE(c.IsMatrix() == false);
            REQUIRE(c.GetDimensions() == nullptr);


            for (auto i = 0; i < size; i++) {
                c.SetVal(i, 3.225);
            }

            for (auto i = 0; i < size; i++) {
                REQUIRE(b.GetVal(i) != c.GetVal(i));
            }

        }
        delete[] validator;
    }

    SECTION("Test Clear Up") {
        DataType temp(30, 1);
        REQUIRE(temp.GetPrecision() == INT);

        temp.ClearUp();
        REQUIRE(temp.GetData() == nullptr);
        REQUIRE(temp.GetDimensions() == nullptr);

    }


}


TEST_CASE("DataTypeTest", "[DataType]") {
    TEST_DATA_TYPE();
}
