

#include <operations/BasicOperations.hpp>
#include <libraries/catch/catch.hpp>
#include <utilities/MPRDispatcher.hpp>


using namespace mpr::operations;
using namespace mpr::precision;
using namespace std;


void
TEST_BASIC_OPERATION() {
    SECTION("Testing Sweep") {
        cout << "Testing Basic Utilities ..." << endl;
        std::cout << "Testing Sweep ..." << endl;

        DataType a(6, 4, FLOAT);
        DataType b(6, FLOAT);
        DataType c(FLOAT);
        int margin = 1;


        auto data_one = (float *) a.GetData();
        auto data_two = (float *) b.GetData();
        auto size = a.GetSize();

        for (auto i = 0; i < size; i++) {
            data_one[ i ] = 0;
        }
        for (auto i = 0; i < a.GetNRow(); i++) {
            data_two[ i ] = i + 1;
        }

        auto validator = new float[size];
        int val;
        auto col = a.GetNCol();
        for (auto i = 0; i < size; i++) {
            val = ( i % col ) + 1;
            validator[ i ] = val;
        }

        DISPATCHER(FFF, basic::Sweep, a, b, c, margin, "+")

        auto temp_out = (float *) c.GetData();
        for (auto i = 0; i < size; i++) {
            REQUIRE(validator[ i ] == temp_out[ i ]);
        }

        delete[] validator;

    }SECTION("Testing Min/Max") {
        cout << "Testing Min/Max With Index ..." << endl;

        DataType a(50, FLOAT);
        DataType output(FLOAT);
        size_t index = 0;

        auto data_in_a = (float *) a.GetData();
        data_in_a[ 20 ] = -15;

        SIMPLE_DISPATCH(FLOAT, basic::MinMax, a, output, index, false)
        auto data_out = (float *) output.GetData();

        REQUIRE(output.GetSize() == 1);
        REQUIRE(index == 20);
        REQUIRE(data_out[ 0 ] == -15);

        data_in_a[ 15 ] = 200;
        index = 0;

        SIMPLE_DISPATCH(FLOAT, basic::MinMax, a, output, index, true)
        data_out = (float *) output.GetData();

        REQUIRE(output.GetSize() == 1);
        REQUIRE(index == 15);
        REQUIRE(data_out[ 0 ] == 200);
    }SECTION("Test Get Diagonal") {
        cout << "Testing Get Diagonal ..." << endl;
        DataType a(5, 5, FLOAT);
        DataType output(FLOAT);

        auto data_in_a = (float *) a.GetData();
        REQUIRE(a.GetDimensions() != nullptr);
        auto num_row = a.GetDimensions()->GetNRow();
        auto num_col = a.GetDimensions()->GetNCol();

        for (auto i = 0; i < num_row; ++i) {
            for (auto j = 0; j < num_col; j++) {
                data_in_a[ ( i * num_col ) + j ] = i;
            }
        }

        SIMPLE_DISPATCH(FLOAT, basic::GetDiagonal, a, output)

        auto data_out = (float *) output.GetData();
        auto size_out = output.GetSize();

        REQUIRE(size_out == 5);
        for (auto i = 0; i < size_out; ++i) {
            REQUIRE(data_out[ i ] == i);
        }


        a.ToVector();
        Dimensions a_dims(5, 5);
        SIMPLE_DISPATCH(FLOAT, basic::GetDiagonal, a, output, &a_dims)

        data_out = (float *) output.GetData();
        size_out = output.GetSize();

        REQUIRE(size_out == 5);
        for (auto i = 0; i < size_out; ++i) {
            REQUIRE(data_out[ i ] == i);
        }


    }SECTION("Test Checking Types") {
        cout << "Testing Type Checks ..." << endl;
        DataType a(FLOAT);
        DataType b(DOUBLE);
        DataType c(INT);

        REQUIRE(basic::IsSFloat(a) == false);
        REQUIRE(basic::IsFloat(a) == true);
        REQUIRE(basic::IsDouble(a) == false);


        REQUIRE(basic::IsSFloat(b) == false);
        REQUIRE(basic::IsFloat(b) == false);
        REQUIRE(basic::IsDouble(b) == true);


        REQUIRE(basic::IsSFloat(c) == true);
        REQUIRE(basic::IsFloat(c) == false);
        REQUIRE(basic::IsDouble(c) == false);


    }SECTION("Testing CBind Same Precision") {
        cout << "Testing CBind ..." << endl;
        DataType a(6, 4, FLOAT);
        DataType b(6, 4, FLOAT);
        DataType c(FLOAT);

        auto data_in_a = (float *) a.GetData();
        auto data_in_b = (float *) b.GetData();

        size_t counter = 0;
        for (auto i = 0; i < 6; ++i) {
            for (auto j = 0; j < 4; j++) {
                data_in_a[ ( i * 4 ) + j ] = counter;
                data_in_b[ ( i * 4 ) + j ] = counter;
                counter++;
            }
        }


        DISPATCHER(FFF, basic::ColumnBind, a, b, c)

        DataType test(6, 8, FLOAT);

        auto size = c.GetSize();
        auto temp_data = (float *) test.GetData();
        auto temp_data_in = (float *) c.GetData();
        REQUIRE(size == ( 6 * 8 ));

        counter = 0;
        for (auto i = 0; i < 6; ++i) {
            for (auto j = 0; j < 4; j++) {
                temp_data[ ( i * 8 ) + j ] = counter;
                temp_data[ (( i * 8 ) + j ) + 4 ] = counter;
                counter++;
            }
        }

        for (auto i = 0; i < size; ++i) {
            REQUIRE(temp_data[ i ] == temp_data_in[ i ]);
        }
    }SECTION("Testing CBind Different Precision") {
        DataType a(6, 4, FLOAT);
        DataType b(6, 4, DOUBLE);
        DataType c(DOUBLE);

        auto data_in_a = (float *) a.GetData();
        auto data_in_b = (double *) b.GetData();

        size_t counter = 0;

        for (auto i = 0; i < 6; ++i) {
            for (auto j = 0; j < 4; j++) {
                data_in_a[ ( i * 4 ) + j ] = counter;
                data_in_b[ ( i * 4 ) + j ] = counter;
                counter++;
            }
        }

        DISPATCHER(FDD, basic::ColumnBind, a, b, c)

        DataType test(6, 8, DOUBLE);
        auto temp_data = (double *) test.GetData();
        auto temp_data_in = (double *) c.GetData();
        auto size = c.GetSize();
        REQUIRE(size == ( 6 * 8 ));

        counter = 0;
        for (auto i = 0; i < 6; ++i) {
            for (auto j = 0; j < 4; j++) {
                temp_data[ ( i * 8 ) + j ] = counter;
                temp_data[ (( i * 8 ) + j ) + 4 ] = counter;
                counter++;
            }
        }

        for (auto i = 0; i < size; ++i) {
            REQUIRE(temp_data[ i ] == temp_data_in[ i ]);
        }

    }SECTION("Testing CBind With Wrong Dimensions") {
        /** TODO: try and catch not working with invalid argument **/
//        DataType a(6, 4, FLOAT);
//        DataType b(3, 4, DOUBLE);
//        DataType c(DOUBLE);
//        try {
//            DISPATCHER(FDD, basic::ColumnBind, a, b, c)
//        } catch (std::invalid_argument const &e) {
//            REQUIRE(e.what() =="");
//        }

    }SECTION("Testing RBind Same Precision") {
        cout << "Testing RBind ..." << endl;

        DataType a(6, 4, FLOAT);
        DataType b(6, 4, FLOAT);
        DataType c(FLOAT);

        auto data_in_a = (float *) a.GetData();
        auto data_in_b = (float *) b.GetData();

        for (auto i = 0; i < a.GetSize(); i++) {
            data_in_a[ i ] = i;
            data_in_b[ i ] = i;
        }


        DISPATCHER(FFF, basic::RowBind, a, b, c)

        DataType test(12, 4, FLOAT);
        auto temp_data = (float *) test.GetData();
        auto temp_data_in = (float *) c.GetData();
        auto size = c.GetSize();
        REQUIRE(size == ( 12 * 4 ));

        for (auto i = 0; i < a.GetSize(); i++) {
            temp_data[ i ] = i;
            temp_data[ i + a.GetSize() ] = i;
        }

        for (auto i = 0; i < size; ++i) {
            REQUIRE(temp_data[ i ] == temp_data_in[ i ]);
        }

    }SECTION("Testing RBind Different Precision") {
        DataType a(6, 4, FLOAT);
        DataType b(6, 4, DOUBLE);
        DataType c(DOUBLE);

        DISPATCHER(FDD, basic::RowBind, a, b, c)

        DataType test(12, 4, DOUBLE);
        auto temp_data = (double *) test.GetData();
        auto temp_data_in = (double *) c.GetData();
        auto size = c.GetSize();

        REQUIRE(size == ( 12 * 4 ));
        for (auto i = 0; i < size; ++i) {
            REQUIRE(temp_data[ i ] == temp_data_in[ i ]);
        }

    }SECTION("Testing RBind With Wrong Dimensions") {
        /** TODO: try and catch not working with invalid argument **/
//        DataType a(6, 4, FLOAT);
//        DataType b(6, 3, DOUBLE);
//        DataType c(DOUBLE);
//        try {
//            DISPATCHER(FDD, basic::RowBind, a, b, c)
//        } catch (std::invalid_argument const &e) {
//            REQUIRE(e.what() == "");
//        }
    }SECTION("Testing Replicate") {
        cout << "Testing Replicate ..." << endl;
        DataType a(5, FLOAT);
        auto data_in_a = (float *) a.GetData();
        auto size_in_a = a.GetSize();

        for (auto i = 0; i < size_in_a; ++i) {
            data_in_a[ i ] = i;
        }

        DataType b(FLOAT);
        SIMPLE_DISPATCH(FLOAT, basic::Replicate, a, b, 50)
        auto size = b.GetSize();
        REQUIRE(size == 50);
        auto data_out = (float *) b.GetData();

        for (auto i = 0; i < size; ++i) {
            REQUIRE(data_out[ i ] == i % 5);
        }
    }SECTION("Test NA Omit") {
        cout << "Testing NA Replace ..." << endl;
        DataType a(50, FLOAT);

        auto data_in_a = (float *) a.GetData();
        auto size = a.GetSize();
        float zero = 0;
        for (auto i = 30; i < size - 10; i++) {
            data_in_a[ i ] = 0;
            data_in_a[ i ] = data_in_a[ i ] / zero;
        }

        SIMPLE_DISPATCH(FLOAT, basic::NAReplace, a, 3.5)

        for (auto i = 30; i < size - 10; i++) {
            REQUIRE(data_in_a[ i ] == 3.5);
        }

        for (auto i = 30; i < size - 10; i++) {
            data_in_a[ i ] = 0;
            data_in_a[ i ] = data_in_a[ i ] / zero;
        }

        cout << "Testing NA Omit ..." << endl;
        SIMPLE_DISPATCH(FLOAT, basic::NAExclude, a)

        data_in_a = (float *) a.GetData();
        size = a.GetSize();
        REQUIRE(size == 40);
        for (auto i = 0; i < size; i++) {
            REQUIRE(data_in_a[ i ] == 1.5);
        }

    }
}


TEST_CASE("BasicOperations", "[BasicOperations]") {
    TEST_BASIC_OPERATION();
}
