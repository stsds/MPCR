/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MMPR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <libraries/catch/catch.hpp>
#include <data-units/DataType.hpp>
#include <utilities/MPRDispatcher.hpp>


using namespace std;
using namespace mpr::precision;


template <typename T>
void
CheckValues(DataType *apData, char *apValidator) {

    T *temp_data = (T *) apData->GetData();
    T *temp_validate = (T *) apValidator;
    auto size = apData->GetSize();
    for (auto i = 0; i < size; i++) {
        REQUIRE(temp_data[ i ] == temp_validate[ i ]);
    }
}


template <typename T>
void
InitValidator(char *&apData, size_t aSize) {
    T *temp = new T[aSize];
    for (auto i = 0; i < aSize; i++) {
        temp[ i ] = (T) 1.5;
    }
    apData = (char *) temp;
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
            REQUIRE(data[ i ] == a.GetVal(i));
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
        REQUIRE(temp.GetPrecision() == HALF);

        temp.ClearUp();
        REQUIRE(temp.GetData() == nullptr);
        REQUIRE(temp.GetDimensions() == nullptr);

    }SECTION("Test Precision Conversion") {

        cout << "Testing Precision Conversion ..." << endl;

        DataType a(35, "float");
        auto pData_in_a = (float *) a.GetData();
        auto size_a = a.GetSize();

        for (auto i = 0; i < size_a; i++) {
            pData_in_a[ i ] = i;
        }

        a.ConvertPrecision(DOUBLE);
        REQUIRE(a.GetSize() == size_a);
        REQUIRE(a.GetPrecision() == DOUBLE);

        auto pData_in_a_new = (double *) a.GetData();
        for (auto i = 0; i < size_a; i++) {
            REQUIRE(pData_in_a_new[ i ] == i);
            pData_in_a_new[ i ] = 1.5;
        }

        a.ConvertPrecision(HALF);

        REQUIRE(a.GetSize() == size_a);
        REQUIRE(a.GetPrecision() == HALF);

        auto pData_in_a_new_int = (int *) a.GetData();
        for (auto i = 0; i < size_a; i++) {
            REQUIRE(pData_in_a_new_int[ i ] == 1);

        }

    }SECTION("Converter") {

        cout << "Testing NumericVector Conversion ..." << endl;

        DataType a(50, FLOAT);
        auto pData_in_a = (float *) a.GetData();
        auto size_a = a.GetSize();

        for (auto i = 0; i < size_a; i++) {
            pData_in_a[ i ] = i;
        }

        size_t i = 0;
        auto pOutput_vector = a.ConvertToNumericVector();
        for (auto x: *pOutput_vector) {
            REQUIRE(x == i);
            i++;
        }

        delete pOutput_vector;

    }SECTION("Test Copy Constructor for different precision") {
        vector <double> vals(50, 5);
        DataType a(vals, DOUBLE);
        DataType b(a, FLOAT);

        REQUIRE(b.GetPrecision() == FLOAT);
        REQUIRE(b.GetSize() == 50);

        for (auto i = 0; i < b.GetSize(); i++) {
            REQUIRE(b.GetVal(i) == 5);
        }
    }SECTION("Test Sum and Product") {
        cout << "Testing MPR Sum ..." << endl;
        vector <double> values;
        auto size = 50;
        values.resize(size);
        double validate_sum = 0;
        double validate_prod = 1;

        for (auto i = 0; i < size; i++) {
            values[ i ] = i + 1;
            validate_sum += values[ i ];
            validate_prod *= values[ i ];
        }


        DataType a(values, FLOAT);
        auto sum = a.Sum();

        REQUIRE(sum == validate_sum);

        cout << "Testing MPR Product ..." << endl;

        double prod = a.Product();
        REQUIRE(prod == validate_prod);

    }SECTION("Testing Determinant") {
        cout << "Testing MPR Determinant ..." << endl;
        vector <double> values{1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1, 2,
                               1, 0, 2, 1, 1, 1, 1, 2, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(5, 5);
        REQUIRE(a.GetSize() == 25);
        REQUIRE(a.Determinant() == 0);

        values = {5, 78, 23, 67, 6, 9, 11, 45, 7, 8, 6, 0, 9, 0, 0, 89};
        DataType b(values, FLOAT);
        b.ToMatrix(4, 4);
        REQUIRE(b.GetSize() == 16);
        double det_act = 374308.9999999999999;
        auto det = b.Determinant();
        auto validate_perc = fabs(det - det_act) / det_act;
        REQUIRE(validate_perc < 0.001);
    }SECTION("Testing Serialization") {
        SECTION("Testing float vec") {
            cout << "Testing Serialization and De-serialization ..." << endl;
            vector <double> values;
            auto size = 50;
            values.resize(size);
            for (auto i = 0; i < size; i++) {
                values[ i ] = i;
            }
            DataType a(values, FLOAT);
            auto temp_val = a.Serialize();
            auto val = temp_val.data();
            auto temp_deserialized = DataType::DeSerialize(val);
            REQUIRE(temp_deserialized->GetSize() == 50);
            REQUIRE(temp_deserialized->IsMatrix() == false);
            REQUIRE(temp_deserialized->GetPrecision() == FLOAT);
            for (auto i = 0; i < size; i++) {
                REQUIRE(temp_deserialized->GetVal(i) == i);
            }

            delete temp_deserialized;
        }SECTION("Testing float matrix") {
            vector <double> values;
            auto size = 50;
            values.resize(size);
            for (auto i = 0; i < size; i++) {
                values[ i ] = i;
            }
            DataType a(values, FLOAT);
            a.ToMatrix(5, 10);
            auto temp_val = a.Serialize();
            auto val = temp_val.data();
            auto temp_deserialized = DataType::DeSerialize(val);
            REQUIRE(temp_deserialized->IsMatrix() == true);
            REQUIRE(temp_deserialized->GetSize() == 50);
            REQUIRE(temp_deserialized->GetNRow() == 5);
            REQUIRE(temp_deserialized->GetNCol() == 10);
            REQUIRE(temp_deserialized->GetPrecision() == FLOAT);
            for (auto i = 0; i < size; i++) {
                REQUIRE(temp_deserialized->GetVal(i) == i);
            }

            delete temp_deserialized;

        }SECTION("Testing double Matrix") {
            vector <double> values;
            auto size = 50;
            values.resize(size);
            for (auto i = 0; i < size; i++) {
                values[ i ] = i;
            }
            DataType a(values, DOUBLE);
            a.ToMatrix(5, 10);
            auto temp_val = a.Serialize();
            auto val = temp_val.data();
            auto temp_deserialized = DataType::DeSerialize(val);
            REQUIRE(temp_deserialized->IsMatrix() == true);
            REQUIRE(temp_deserialized->GetSize() == 50);
            REQUIRE(temp_deserialized->GetNRow() == 5);
            REQUIRE(temp_deserialized->GetNCol() == 10);
            REQUIRE(temp_deserialized->GetPrecision() == DOUBLE);
            for (auto i = 0; i < size; i++) {
                REQUIRE(temp_deserialized->GetVal(i) == i);
            }

            delete temp_deserialized;

        }


    }


}


TEST_CASE("DataTypeTest", "[DataType]") {
    TEST_DATA_TYPE();
}
