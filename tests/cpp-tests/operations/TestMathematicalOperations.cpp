
#include <libraries/catch/catch.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <operations/MathematicalOperations.hpp>


using namespace std;
using namespace mpr::precision;
using namespace mpr::operations;


void
TEST_MATH_OPERATIONS() {
    SECTION("Test Round") {
        cout << "Testing Round Operation ..." << endl;
        vector <double> values = {1.333, 2.25, 3.111111, 4.555};
        DataType a(values, FLOAT);
        DataType out(FLOAT);

        SIMPLE_DISPATCH(FLOAT, math::PerformRoundOperation, a, out, "trunc")

        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            REQUIRE(out.GetVal(i) == i + 1);
        }
    }SECTION("Test abs") {

        cout << "Testing Abs Operation ..." << endl;
        vector <double> values = {-1.333, 2.25, -3.111111, 4.555};
        DataType a(values, FLOAT);
        DataType out(FLOAT);

        SIMPLE_DISPATCH(FLOAT, math::PerformRoundOperation, a, out, "abs")

        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            REQUIRE(out.GetVal(i) == std::abs((float) values[ i ]));
        }

    }SECTION("Test floor") {

        cout << "Testing Floor Operation ..." << endl;
        vector <double> values = {-1.333, 2.25, -3.111111, 4.555};
        DataType a(values, FLOAT);
        DataType out(FLOAT);

        SIMPLE_DISPATCH(FLOAT, math::PerformRoundOperation, a, out, "floor")

        values.clear();
        values = {-2, 2, -4, 4};
        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            REQUIRE(out.GetVal(i) == values[ i ]);
        }

    }SECTION("Test ceil") {
        cout << "Testing Ceil Operation ..." << endl;
        vector <double> values = {-1.333, 2.25, -3.111111, 4.555};
        DataType a(values, FLOAT);
        DataType out(FLOAT);

        SIMPLE_DISPATCH(FLOAT, math::PerformRoundOperation, a, out, "ceil")

        values.clear();
        values = {-1, 3, -3, 5};
        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            REQUIRE(out.GetVal(i) == values[ i ]);
        }
    }SECTION("Test Round Decimal Point") {

        cout << "Testing Round With Decimal Point Operation ..." << endl;
        vector <double> values = {-1.333, 2.25, -3.111111, 4.555};
        DataType a(values, FLOAT);
        DataType out(FLOAT);

        SIMPLE_DISPATCH(FLOAT, math::Round, a, out, 2)

        values.clear();
        values = {-1.33, 2.25, -3.11, 4.55};
        auto error = 0.001;
        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - (float) values[ i ]) /
                       (float) values[ i ];
            REQUIRE(val <= error);
        }
    }SECTION("Test Trig Operations") {

        cout << "Testing Cos ..." << endl;
        vector <double> values = {-1.333, 2.25, -3.111111, 4.555};
        DataType a(values, FLOAT);
        DataType out(FLOAT);

        SIMPLE_DISPATCH(FLOAT, math::PerformTrigOperation, a, out, "cos")

        auto error = 0.001;
        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - cos(values[ i ])) /
                       cos((float) values[ i ]);
            REQUIRE(val <= error);
        }

        cout << "Testing Sin ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformTrigOperation, a, out, "sin")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - sin(values[ i ])) /
                       sin((float) values[ i ]);
            REQUIRE(val <= error);
        }

        cout << "Testing Tan ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformTrigOperation, a, out, "tan")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - tan(values[ i ])) /
                       tan((float) values[ i ]);
            REQUIRE(val <= error);
        }

        cout << "Testing Cosh ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformTrigOperation, a, out, "cosh")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - cosh(values[ i ])) /
                       cosh((float) values[ i ]);
            REQUIRE(val <= error);
        }

        cout << "Testing Sinh ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformTrigOperation, a, out, "sinh")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - sinh(values[ i ])) /
                       sinh((float) values[ i ]);
            REQUIRE(val <= error);
        }

        cout << "Testing Tanh ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformTrigOperation, a, out, "tanh")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - tanh(values[ i ])) /
                       tanh((float) values[ i ]);
            REQUIRE(val <= error);
        }


        cout << "Testing acos ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformInverseTrigOperation, a, out,
                        "acos")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - acos(values[ i ])) /
                       acos((float) values[ i ]);
            if (!isnan(val)) {
                REQUIRE(val <= error);
            }
        }

        cout << "Testing asin ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformInverseTrigOperation, a, out,
                        "asin")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - asin(values[ i ])) /
                       asin((float) values[ i ]);
            if (!isnan(val)) {
                REQUIRE(val <= error);
            }
        }

        cout << "Testing atan ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformInverseTrigOperation, a, out,
                        "atan")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - atan(values[ i ])) /
                       atan((float) values[ i ]);
            if (!isnan(val)) {
                REQUIRE(val <= error);
            }
        }

        cout << "Testing acosh ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformInverseTrigOperation, a, out,
                        "acosh")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - acosh(values[ i ])) /
                       acosh((float) values[ i ]);
            if (!isnan(val)) {
                REQUIRE(val <= error);
            }
        }

        cout << "Testing asinh ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformInverseTrigOperation, a, out,
                        "asinh")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - asinh(values[ i ])) /
                       asinh((float) values[ i ]);
            if (!isnan(val)) {
                REQUIRE(val <= error);
            }
        }

        cout << "Testing atanh ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::PerformInverseTrigOperation, a, out,
                        "atanh")


        REQUIRE(out.GetSize() == 4);
        for (auto i = 0; i < out.GetSize(); i++) {
            auto val = fabs((float) out.GetVal(i) - atanh(values[ i ])) /
                       atanh((float) values[ i ]);
            if (!isnan(val)) {
                REQUIRE(val <= error);
            }
        }

    }SECTION("Test Finite") {

        cout << "Testing Finite ..." << endl;
        vector <double> values = {-1.333, 2.25, -3.111111, 4.555};
        DataType a(values, FLOAT);
        a.SetVal(2, NAN);
        vector <int> output;

        SIMPLE_DISPATCH(FLOAT, math::IsFinite, a, output)
        REQUIRE(output.size() == 4);
        REQUIRE(output[ 2 ] == 0);
        for (auto i = 0; i < output.size(); i++) {
            if (i != 2) {
                REQUIRE(output[ i ] == 1);
            }
        }

        cout << "Testing IsInFinite ..." << endl;

        vector <int> out;
        a.SetVal(1, INFINITY);
        SIMPLE_DISPATCH(FLOAT, math::IsInFinite, a, out)

        REQUIRE(out.size() == 4);
        REQUIRE(out[ 2 ] == INT_MIN);
        REQUIRE(out[ 1 ] == true);
        for (auto i = 0; i < output.size(); i++) {
            if (i == 2 || i == 1) {
                continue;
            } else {
                REQUIRE(out[ i ] == false);
            }
        }
    }SECTION("Testing Log") {
        cout << "Testing Log 10 ..." << endl;
        vector <double> values = {1.333, 2.25, 3.111111, 4.555};
        DataType a(values, FLOAT);
        DataType b(FLOAT);

        SIMPLE_DISPATCH(FLOAT, math::Log, a, b, 10)
        REQUIRE(b.GetSize() == 4);

        for (auto i = 0; i < b.GetSize(); i++) {
            REQUIRE(b.GetVal(i) == log10((float) values[ i ]));
        }

        cout << "Testing Log 2 ..." << endl;
        SIMPLE_DISPATCH(FLOAT, math::Log, a, b, 2)
        REQUIRE(b.GetSize() == 4);

        for (auto i = 0; i < b.GetSize(); i++) {
            REQUIRE(b.GetVal(i) == log2((float) values[ i ]));
        }

    }SECTION("Test Exponential") {

        cout << "Testing Exponential ..." << endl;
        vector <double> values = {1.333, 2.25, 3.111111, 4.555};
        DataType a(values, FLOAT);
        DataType b(FLOAT);

        SIMPLE_DISPATCH(FLOAT, math::Exponential, a, b, false)
        REQUIRE(b.GetSize() == 4);

        for (auto i = 0; i < b.GetSize(); i++) {
            REQUIRE(b.GetVal(i) == exp((float) values[ i ]));
        }

    }SECTION("Test SquareRoot") {

        cout << "Testing Exponential ..." << endl;
        vector <double> values = {1, 4, 9, 16, 25, 36};
        DataType a(values, FLOAT);
        DataType b(FLOAT);

        SIMPLE_DISPATCH(FLOAT, math::SquareRoot, a, b)
        REQUIRE(b.GetSize() == 6);

        for (auto i = 0; i < b.GetSize(); i++) {
            REQUIRE(b.GetVal(i) == i + 1);
        }

    }SECTION("Test Gamma") {
        cout << "Testing Gamma ..." << endl;
        vector <double> values = {1, 4, 9, 16, 25, 36};
        DataType a(values, FLOAT);
        DataType b(FLOAT);

        SIMPLE_DISPATCH(FLOAT, math::Gamma, a, b, false)
        REQUIRE(b.GetSize() == 6);

        auto error = 0.001;
        for (auto i = 0; i < b.GetSize(); i++) {
            auto val = fabs((float) b.GetVal(i) - (float) tgamma(values[ i ])) /
                       (float) tgamma(values[ i ]);
            if (!isnan(val)) {
                REQUIRE(val <= error);
            }
        }

        SIMPLE_DISPATCH(FLOAT, math::Gamma, a, b, true)
        REQUIRE(b.GetSize() == 6);

        for (auto i = 0; i < b.GetSize(); i++) {
            auto val = fabs((float) b.GetVal(i) - (float) lgamma(values[ i ])) /
                       (float) lgamma(values[ i ]);
            if (!isnan(val)) {
                REQUIRE(val <= error);
            }
        }

    }
}


TEST_CASE("Math Operations", "[Math Operations]") {
    TEST_MATH_OPERATIONS();
}
