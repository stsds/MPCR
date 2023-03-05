
#include <operations/BinaryOperations.hpp>
#include <libraries/catch/catch.hpp>
#include <utilities/MPRDispatcher.hpp>


using namespace mpr::operations;
using namespace mpr::precision;
using namespace std;


void
TEST_BINARY_OPERATION() {
    SECTION("Test Addition") {
        cout << "-----------------------------------" << endl;
        cout << "Testing Basic Arithmetic Operations ..." << endl;
        cout << "Testing Between 2 MPR Objects ..." << endl;


        DataType a(50, FLOAT);
        DataType b(30, FLOAT);
        DataType output(FLOAT);

        auto pData_in_a = (float *) a.GetData();
        auto pData_in_b = (float *) b.GetData();
        auto size_in_a = a.GetSize();
        auto size_in_b = b.GetSize();


        for (auto i = 0; i < size_in_a; i++) {
            pData_in_a[ i ] = i + 1;
        }

        for (auto i = 0; i < size_in_b; i++) {
            pData_in_b[ i ] = i + 1;
        }


        a.ToVector();
        b.ToVector();

        DISPATCHER(FFF, binary::PerformOperation, a, b, output, "+")


        auto pData_out = (float *) output.GetData();
        auto size = output.GetSize();
        REQUIRE(size == 50);

        for (auto i = 0; i < size; i++) {
            REQUIRE(pData_out[ i ] ==
                    pData_in_a[ i ] + pData_in_b[ i % size_in_b ]);
        }


        DISPATCHER(FFF, binary::PerformOperation, b, a, output, "+")

        pData_out = (float *) output.GetData();
        size = output.GetSize();
        REQUIRE(size == 50);

        for (auto i = 0; i < size; i++) {
            REQUIRE(pData_out[ i ] ==
                    pData_in_a[ i ] + pData_in_b[ i % size_in_b ]);
        }

        DataType d(50, FLOAT);

        auto pData_in_d = (float *) d.GetData();
        auto size_in_d = d.GetSize();

        for (auto i = 0; i < size_in_d; i++) {
            pData_in_d[ i ] = 2;
        }

        DISPATCHER(FFF, binary::PerformOperation, d, a, output, "^")

        pData_out = (float *) output.GetData();
        size = output.GetSize();
        REQUIRE(size == 50);

        size_t val = 2;
        for (auto i = 0; i < size; i++) {
            REQUIRE(pData_out[ i ] == val);
            val = val << 1;
        }

        cout << "Testing Between a Value and MPR object" << endl;
        cout << "-----------------------------------" << endl;
        auto aVal = 5;
        DISPATCHER(FFF, binary::PerformOperationSingle, a, aVal, output, "^")

        pData_out = (float *) output.GetData();
        size = output.GetSize();
        REQUIRE(size == 50);

        for (auto i = 0; i < size; i++) {
            val = pow(i + 1, aVal);
            REQUIRE(pData_out[ i ] == val);
        }

        cout << "Testing Between a MPR Vector and MPR Matrix" << endl;
        cout << "-----------------------------------" << endl;
        DataType x(5, 10, FLOAT);
        DataType y(50, FLOAT);
        DataType output_matrix(FLOAT);


        auto col = x.GetNCol();
        auto row = x.GetNRow();

        auto pData_in_x = (float *) x.GetData();
        auto pData_in_y = (float *) y.GetData();

        auto size_in_x = x.GetSize();

        for (auto i = 0; i < size_in_x; i++) {
            pData_in_x[ i ] = i + 1;
            pData_in_y[ i ] = i + 1;
        }


        DISPATCHER(FFF, binary::PerformOperation, x, y, output_matrix, "+")

        pData_out = (float *) output_matrix.GetData();
        size = output_matrix.GetSize();
        auto matrix_col = output_matrix.GetNCol();
        auto matrix_row = output_matrix.GetNRow();
        REQUIRE(matrix_col == 10);
        REQUIRE(matrix_row == 5);
        REQUIRE(size == 50);
        auto counter = 1;
        auto pValidator = new float[50];
        for (auto i = 0; i < matrix_col; i++) {
            auto start_idx = 0;
            for (auto j = 0; j < matrix_row; j++) {
                start_idx = ( i * matrix_row ) + j;
                pValidator[ start_idx ] = counter;
                counter++;
            }
        }

        for (auto i = 0; i < size; i++) {
            REQUIRE(pData_out[ i ] == pValidator[ i ] + pData_in_y[ i ]);
        }

        delete[] pValidator;

    }SECTION("Testing Binary Comparisons") {
        cout << "Testing Binary Comparisons ... " << endl;

        DataType vals_a(50, "float");
        DataType vals_b(50, "double");

        auto pData_in_vals_a = (float *) vals_a.GetData();
        auto pData_in_vals_b = (float *) vals_b.GetData();

        auto size_in_vals_a = vals_a.GetSize();

        for (auto i = 0; i < size_in_vals_a; i++) {
            pData_in_vals_a[ i ] = i;
            pData_in_vals_b[ i ] = i + size_in_vals_a;
        }

        vector <int> compare_output;
        Dimensions *pTemp_dims = nullptr;
        DISPATCHER(FFF, binary::PerformCompareOperation, vals_a, vals_b,
                   compare_output, ">", pTemp_dims)


        REQUIRE(compare_output.size() == 50);
        for (auto x: compare_output) {
            REQUIRE(x == 0);
        }
        REQUIRE(pTemp_dims == nullptr);
        compare_output.clear();

        DISPATCHER(FFF, binary::PerformCompareOperation, vals_a, vals_b,
                   compare_output, "<", pTemp_dims)

        REQUIRE(compare_output.size() == 50);
        for (auto x: compare_output) {
            REQUIRE(x == 1);
        }
        REQUIRE(pTemp_dims == nullptr);

        compare_output.clear();
        DISPATCHER(FFF, binary::PerformEqualityOperation, vals_a, vals_b,
                   compare_output, false, pTemp_dims)

        REQUIRE(compare_output.size() == 50);
        for (auto x: compare_output) {
            REQUIRE(x == 0);
        }
        REQUIRE(pTemp_dims == nullptr);

        compare_output.clear();
        DISPATCHER(FFF, binary::PerformEqualityOperation, vals_a, vals_b,
                   compare_output, true, pTemp_dims)

        REQUIRE(compare_output.size() == 50);
        for (auto x: compare_output) {
            REQUIRE(x == 1);
        }
        REQUIRE(pTemp_dims == nullptr);

        for (auto i = 0; i < size_in_vals_a; i++) {
            pData_in_vals_b[ i ] = i;
        }

        compare_output.clear();
        DISPATCHER(FFF, binary::PerformEqualityOperation, vals_a, vals_b,
                   compare_output, false, pTemp_dims)

        REQUIRE(compare_output.size() == 50);
        for (auto x: compare_output) {
            REQUIRE(x == 1);
        }
        REQUIRE(pTemp_dims == nullptr);


        compare_output.clear();
        DISPATCHER(FFF, binary::PerformEqualityOperation, vals_a, vals_b,
                   compare_output, true, pTemp_dims)

        REQUIRE(compare_output.size() == 50);
        for (auto x: compare_output) {
            REQUIRE(x == 0);
        }
        REQUIRE(pTemp_dims == nullptr);

    }SECTION("Test Size Recycle") {
        DataType a(10, FLOAT);
        DataType b(50, FLOAT);
        DataType output(FLOAT);

        auto size_a = a.GetSize();
        auto size_b = b.GetSize();

        for (auto i = 0; i < size_a; i++) {
            a.SetVal(i, i + 1);
        }

        for (auto i = 0; i < size_b; i++) {
            b.SetVal(i, i + 1);
        }

        DISPATCHER(FFF, binary::PerformOperation, a, b, output, "*")

        auto size_out = output.GetSize();
        REQUIRE(size_out == 50);

        for (auto i = 0; i < size_out; i++) {
            REQUIRE(output.GetVal(i) ==
                    a.GetVal(i % size_a) * b.GetVal(i % size_b));
        }

        vector <int> output_operations;
        Dimensions *temp = nullptr;

        DISPATCHER(FFF, binary::PerformCompareOperation, a, b,
                   output_operations, ">", temp)

        REQUIRE(output_operations.size() == 50);
        REQUIRE(temp == nullptr);
        auto i = 0;
        for (auto &x: output_operations) {
            REQUIRE(x == ( a.GetVal(i % size_a) > b.GetVal(i % size_b)));
            i++;
        }

        output_operations.clear();

        DISPATCHER(FFF, binary::PerformEqualityOperation, a, b,
                   output_operations, false, temp)

        REQUIRE(output_operations.size() == 50);
        REQUIRE(temp == nullptr);
        i = 0;
        for (auto &x: output_operations) {
            REQUIRE(x == ( a.GetVal(i % size_a) == b.GetVal(i % size_b)));
            i++;
        }

    }
}


TEST_CASE("BinaryOperations", "[BinaryOperations]") {
    TEST_BINARY_OPERATION();
}
