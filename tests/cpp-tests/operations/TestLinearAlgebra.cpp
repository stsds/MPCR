/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <libraries/catch/catch.hpp>
#include <utilities/MPCRDispatcher.hpp>
#include <operations/LinearAlgebra.hpp>
#include <operations/MathematicalOperations.hpp>


using namespace std;
using namespace mpcr::precision;
using namespace mpcr::operations;


void
TEST_LINEAR_ALGEBRA() {
    SECTION("Test CrossProduct") {
        vector<double> values = {3.12393, -1.16854, -0.304408, -2.15901,
                                 -1.16854, 1.86968, 1.04094, 1.35925,
                                 -0.304408, 1.04094, 4.43374, 1.21072,
                                 -2.15901, 1.35925, 1.21072, 5.57265};

        DataType a(values, DOUBLE);
        a.ToMatrix(4, 4);

        DataType b(values, DOUBLE);
        b.ToMatrix(4, 4);
        DataType output(DOUBLE);

        vector<double> validate_vals = {15.878412787064, -9.08673783542,
                                        -6.13095182416, -20.73289403456,
                                        -9.08673783542, 7.7923056801,
                                        8.56286609912, 13.8991634747,
                                        -6.13095182416, 8.56286609912,
                                        22.300113620064, 14.18705411188,
                                        -20.73289403456, 13.8991634747,
                                        14.18705411188, 39.0291556835};

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output, false,
                        false)
        REQUIRE(output.GetNRow() == 4);
        REQUIRE(output.GetNCol() == 4);

        auto error = 0.001;
        for (auto i = 0; i < validate_vals.size(); i++) {
            auto val =
                    fabs((float) output.GetVal(i) - (float) validate_vals[i]) /
                    (float) validate_vals[i];
            REQUIRE(val <= error);
        }

        values.clear();
        values = {5, 11, 143, 10, 123, 132};
        a.ClearUp();
        a.Allocate(values);
        a.ToMatrix(3, 2);


        values.clear();
        values = {2, 3, 5, 6, 8, 11, 13, 14, 20, 30};
        b.ClearUp();
        b.Allocate(values);
        b.ToMatrix(5, 2);

        output.ClearUp();

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output, false,
                        true)

        REQUIRE(output.GetNRow() == 3);
        REQUIRE(output.GetNCol() == 5);

        validate_vals.clear();
        validate_vals = {120, 1375, 1738, 145, 1632,
                         2145, 165, 1777, 2563, 230,
                         2525, 3498, 340, 3778, 5104};

        for (auto i = 0; i < validate_vals.size(); i++) {
            auto val =
                    fabs((float) output.GetVal(i) - (float) validate_vals[i]) /
                    (float) validate_vals[i];
            REQUIRE(val <= error);
        }

        values.clear();
        values = {2, 3, 5, 6, 8, 11, 13, 14, 20, 30};

        DataType c(values, FLOAT);
        c.ToMatrix(5, 2);

        DataType d(0, FLOAT);

        output.ClearUp();
        output.ConvertPrecision(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::CrossProduct, c, d, output, false,
                        true)

        validate_vals.clear();
        validate_vals = {125, 149, 164, 232, 346, 149, 178, 197, 278, 414, 164,
                         197, 221, 310, 460, 232, 278, 310, 436, 648, 346, 414,
                         460, 648, 964};

        for (auto i = 0; i < output.GetSize(); i++) {
            REQUIRE(output.GetVal(i) == validate_vals[i]);
        }


    }SECTION("Test Symmetric") {
        cout << "Testing Matrix Is Symmetric ..." << endl;
        vector<double> values = {2, 3, 6, 3, 4, 5, 6, 5, 9};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);
        auto isSymmetric = false;
        SIMPLE_DISPATCH(FLOAT, linear::IsSymmetric, a, isSymmetric)
        REQUIRE(isSymmetric == true);

        values.clear();
        values = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        for (auto i = 0; i < values.size(); i++) {
            a.SetVal(i, values[i]);
        }


        isSymmetric = true;
        SIMPLE_DISPATCH(FLOAT, linear::IsSymmetric, a, isSymmetric)
        REQUIRE(isSymmetric == false);

    }SECTION("Testing Transpose") {
        cout << "Testing Matrix Transpose ..." << endl;
        vector<double> values = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 4);
        a.Transpose();


        REQUIRE(a.GetNRow() == 4);
        REQUIRE(a.GetNCol() == 3);
        for (auto i = 0; i < a.GetSize(); i++) {
            REQUIRE(a.GetVal(i) == i + 1);
        }

    }SECTION("Testing Cholesky Decomposition") {
        cout << "Testing Cholesky Decomposition ..." << endl;
        vector<double> values = {4, 12, -16, 12, 37, -43, -16, -43, 98};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);
        DataType b(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Cholesky, a, b)

        vector<double> values_validate = {2, 0, 0, 6, 1, 0, -8, 5, 3};

        REQUIRE(b.GetNCol() == 3);
        REQUIRE(b.GetNRow() == 3);
        for (auto i = 0; i < b.GetSize(); i++) {
            REQUIRE(b.GetVal(i) == values_validate[i]);
        }

    }SECTION("Test Cholesky Inverse ") {
        cout << "Testing Cholesky Inverse ..." << endl;

        vector<double> values = {1, 0, 0, 1, 1, 0, 1, 2, 1.414214};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        DataType b(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::CholeskyInv, a, b, a.GetNCol())
        vector<float> values_validate = {2.5, -2.0, 0.5, -2, 3, -1, 0.5, -1.0,
                                         0.5};
        REQUIRE(b.GetNCol() == 3);
        REQUIRE(b.GetNRow() == 3);

        float error = 0.001;
        for (auto i = 0; i < b.GetSize(); i++) {
            float val =
                    fabs((float) b.GetVal(i) - (float) values_validate[i]) /
                    (float) values_validate[i];
            REQUIRE(val <= error);
        }

        SIMPLE_DISPATCH(FLOAT, linear::CholeskyInv, a, b, 2)
        values_validate = {2, -1, -1, 1};

        REQUIRE(b.GetNCol() == 2);
        REQUIRE(b.GetNRow() == 2);


        for (auto i = 0; i < b.GetSize(); i++) {
            auto val =
                    fabs((float) b.GetVal(i) - (float) values_validate[i]) /
                    (float) values_validate[i];
            REQUIRE(val <= error);
        }
    }SECTION("Testing Solve Two Input") {
        cout << "Testing Solve Two Input ..." << endl;
        vector<double> values = {3, 1, 4, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(2, 2);

        values.clear();
        values = {10, 4};
        DataType b(values, FLOAT);
        b.ToMatrix(2, 1);

        DataType output(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Solve, a, b, output, false)

        values.clear();
        values = {6, -2};

        REQUIRE(output.GetNCol() == 1);
        REQUIRE(output.GetNRow() == 2);

        float error = 0.001;
        for (auto i = 0; i < output.GetSize(); i++) {
            auto val = fabs((float) output.GetVal(i) - (float) values[i]) /
                       (float) values[i];
            REQUIRE(val <= error);
        }

    }SECTION("Testing Solve One Input") {
        cout << "Testing Solve One Input ..." << endl;
        vector<double> values = {3, 1, 4, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(2, 2);

        DataType b(FLOAT);
        DataType output(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Solve, a, b, output, true)

        vector<double> validate_vals = {-1, 1, 4, -3};


        REQUIRE(output.GetNCol() == 2);
        REQUIRE(output.GetNRow() == 2);

        float error = 0.001;
        for (auto i = 0; i < validate_vals.size(); i++) {
            auto val =
                    fabs((float) output.GetVal(i) - (float) validate_vals[i]) /
                    (float) validate_vals[i];
            REQUIRE(val <= error);
        }

    }SECTION("Testing Back solve") {
        cout << "Testing Back Solve ..." << endl;
        vector<double> values = {1, 0, 0, 2, 1, 0, 3, 1, 2};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        values.clear();
        values = {8, 4, 2};
        DataType b(values, FLOAT);
        b.ToMatrix(3, 1);
        DataType c(FLOAT);


        SIMPLE_DISPATCH(FLOAT, linear::BackSolve, a, b, c, a.GetNCol(), true,
                        false)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        vector<double> validate = {-1, 3, 1};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[i]);
        }

        c.ClearUp();
        SIMPLE_DISPATCH(FLOAT, linear::BackSolve, a, b, c, a.GetNCol(), true,
                        true)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        validate.clear();
        validate = {8, -12, -5};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[i]);
        }
    }SECTION("Testing Forward Solve") {
        cout << "Testing Forward Solve ..." << endl;

        vector<double> values = {1, 0, 0, 2, 1, 0, 3, 1, 2};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        values.clear();
        values = {8, 4, 2};
        DataType b(values, FLOAT);
        b.ToMatrix(3, 1);
        DataType c(FLOAT);


        SIMPLE_DISPATCH(FLOAT, linear::BackSolve, a, b, c, a.GetNCol(), false,
                        false)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        vector<double> validate = {8, 4, 1};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[i]);
        }

        c.ClearUp();
        SIMPLE_DISPATCH(FLOAT, linear::BackSolve, a, b, c, a.GetNCol(), false,
                        true)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        validate.clear();
        validate = {8, 4, 1};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[i]);
        }

    }SECTION("SVD") {
        cout << "Testing Singular Value Decomposition ..." << endl;

        vector<double> values = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 1, 1, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(9, 4);

        vector<float> validate_values = {3.464102e+00, 1.732051e+00,
                                         1.732051e+00, 1.922963e-16};

        DataType d(FLOAT);
        DataType u(FLOAT);
        DataType v(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::SVD, a, d, u, v, a.GetNCol(),
                        a.GetNCol())
        REQUIRE(d.GetSize() == 4);
        auto err = 0.001;

        DataType dd(9, 4, FLOAT);


        for (auto i = 0; i < dd.GetSize(); i++) {
            dd.SetVal(i, 0);
        }

        for (auto i = 0; i < 4; i++) {
            dd.SetValMatrix(i, i, d.GetVal(i));
        }

        vector<double> temp_vals(81, 0);
        DataType uu(temp_vals, FLOAT);
        uu.ToMatrix(9, 9);

        for (auto i = 0; i < u.GetSize(); i++) {
            uu.SetVal(i, u.GetVal(i));
        }

        DataType vv = v;
        vv.Transpose();


        DataType temp_one(FLOAT);
        DataType temp_two(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::CrossProduct, uu, dd, temp_one, false,
                        false)

        SIMPLE_DISPATCH(FLOAT, linear::CrossProduct, temp_one, vv, temp_two,
                        false,
                        false)

        DataType temp_three(FLOAT);
        SIMPLE_DISPATCH(FLOAT, math::Round, temp_two, temp_three, 1);

        SIMPLE_DISPATCH(FLOAT, math::PerformRoundOperation, temp_three,
                        temp_two, "abs");

        for (auto i = 0; i < a.GetSize(); i++) {
            REQUIRE(temp_two.GetVal(i) == a.GetVal(i));
        }


    }SECTION("Eigen") {
        cout << "Testing Eigen ..." << endl;

        vector<double> values = {1, -1, -1, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(2, 2);


        DataType vals(FLOAT);
        DataType vec(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Eigen, a, vals, &vec)

        REQUIRE(vals.GetSize() == 2);
        REQUIRE(vals.GetVal(0) == 2);
        REQUIRE(vals.GetVal(1) == 0);

        REQUIRE(vec.GetSize() == 4);
        REQUIRE(vec.GetNCol() == 2);
        REQUIRE(vec.GetNRow() == 2);


        vector<float> validate_values = {-0.7071068, 0.7071068, -0.7071068,
                                         -0.7071068};
        auto err = 0.001;
        for (auto i = 0; i < vec.GetSize(); i++) {
            auto val =
                    fabs((float) vec.GetVal(i) - (float) validate_values[i]) /
                    (float) validate_values[i];
            REQUIRE(val <= err);
        }
    }SECTION("Norm") {
        cout << "Testing Norm ..." << endl;

        vector<double> values = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5,
                                 6, 7, 8, 9, 10};
        DataType a(values, FLOAT);
        a.ToMatrix(10, 2);

        double norm_val = 0;

        SIMPLE_DISPATCH(FLOAT, linear::Norm, a, "O", norm_val)

        REQUIRE(norm_val == 55);

        SIMPLE_DISPATCH(FLOAT, linear::Norm, a, "I", norm_val)

        REQUIRE(norm_val == 11);

        SIMPLE_DISPATCH(FLOAT, linear::Norm, a, "M", norm_val)

        REQUIRE(norm_val == 10);

        SIMPLE_DISPATCH(FLOAT, linear::Norm, a, "F", norm_val)

        auto val = fabs(norm_val - 19.87461) / 19.87461;
        REQUIRE(val <= 0.001);

    }SECTION("QR Decomposition") {
        cout << "Testing QR Decomposition ..." << endl;
        vector<double> values = {1, 2, 3, 2, 4, 6, 3, 3, 3};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        DataType qraux(FLOAT);
        DataType pivot(FLOAT);
        DataType qr(FLOAT);
        DataType rank(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::QRDecomposition, a, qr, qraux, pivot,
                        rank)

        vector<float> validate_vals = {-7.48331, 0.42179, 0.63269, -4.81070,
                                       1.96396, 0.85977, -3.7417, 0, 0};


        REQUIRE(qr.IsMatrix());
        REQUIRE(qr.GetNCol() == 3);
        REQUIRE(qr.GetNRow() == 3);

        auto err = 0.001;
        for (auto i = 0; i < qr.GetSize(); i++) {

            if (validate_vals[i] != 0) {
                auto val =
                        fabs(qr.GetVal(i) - validate_vals[i]) /
                        validate_vals[i];
                REQUIRE(val <= err);
            } else {
                REQUIRE(qr.GetVal(i) <= 1e-07);
            }

        }

        REQUIRE(rank.GetVal(0) == 2);

        validate_vals.clear();
        validate_vals = {1.2673, 1.1500, 0.0000};
        REQUIRE(qraux.GetSize() == 3);

        for (auto i = 0; i < 2; i++) {
            auto val =
                    fabs(qraux.GetVal(i) - validate_vals[i]) / validate_vals[i];
            REQUIRE(val <= err);
        }

        REQUIRE(qraux.GetVal(2) == 0);

        validate_vals.clear();
        validate_vals = {2, 3, 1};
        for (auto i = 0; i < pivot.GetSize(); i++) {
            REQUIRE(pivot.GetVal(i) == validate_vals[i]);
        }

    }SECTION("Test QR.R") {

        cout << "Testing QR Auxiliaries ..." << endl;
        cout << "Testing QR.R ..." << endl;


        vector<double> values = {-7.48331, 0.42179, 0.63269, -4.81070, 1.96396,
                                 0.85977, -3.7417, 0, 0};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);
        DataType r(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::QRDecompositionR, a, r, true)

        vector<float> validate_vals = {-7.4833, 0, 0, -4.8107, 1.9640, 0,
                                       -3.7417, 0, 0};
        auto error = 0.001;
        for (auto i = 0; i < r.GetSize(); i++) {
            if (validate_vals[i] != 0) {
                auto val =
                        fabs(r.GetVal(i) - validate_vals[i]) / validate_vals[i];
                REQUIRE(val <= error);
            }
        }

    }SECTION("Test QR.Q") {

        cout << "Testing QR.Q ..." << endl;

        vector<double> values = {1, 2, 3, 2, 5, 6, 3, 3, 3};
        DataType a(values, DOUBLE);
        a.ToMatrix(3, 3);


        DataType qraux(DOUBLE);
        DataType pivot(DOUBLE);
        DataType qr(DOUBLE);
        DataType rank(DOUBLE);

        SIMPLE_DISPATCH(DOUBLE, linear::QRDecomposition, a, qr, qraux, pivot,
                        rank)


        DataType r(DOUBLE);
        SIMPLE_DISPATCH(DOUBLE, linear::QRDecompositionR, qr, r, true)

        DataType q(DOUBLE);
        SIMPLE_DISPATCH(DOUBLE, linear::QRDecompositionQ, qr, qraux, q, true)

        DataType output(DOUBLE);
        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, q, r, output, false,
                        false)


        DataType output_temp(DOUBLE);
        SIMPLE_DISPATCH(DOUBLE, math::Round, output, output_temp, 1);

        values.clear();
        values = {2, 5, 6, 3, 3, 3, 1, 2, 3};

        for (auto i = 0; i < output.GetSize(); i++) {
            REQUIRE(output_temp.GetVal(i) == values[i]);
        }


    }SECTION("Testing R Cond") {

        cout << "Testing R Cond ..." << endl;

        vector<double> values = {100, 2, 3, 3, 2, 1, 300, 3, 3, 400, 5, 6, 4,
                                 44, 56, 1223};
        DataType a(values, FLOAT);
        a.ToMatrix(4, 4);

        double b = 0;


        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "I", false)
        auto val = fabs(b - 0.079608) / 0.079608;
        REQUIRE(val <= 0.001);

        b = 0;

        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "O", false)
        val = fabs(b - 0.074096) / 0.074096;
        REQUIRE(val <= 0.001);

        b = 0;

        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "I", true)
        val = fabs(b - 1.3189e-05) / 1.3189e-05;

        REQUIRE(val <= 0.001);
        b = 0;

        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "O", true)

        val = fabs(b - 1.334e-05) / 1.334e-05;
        REQUIRE(val <= 0.001);
    }SECTION("Testing Trmm") {
        cout << "Testing Trmm ..." << endl;
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                CPU);

        // Test upper triangle, not transpose and A on the left side
        vector<double> values = {1, 0, 0, 2, 1, 0, 3, 1, 2};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        values.clear();
        values = {8, 4, 2};
        DataType b(values, FLOAT);
        b.ToMatrix(3, 1);
        DataType c(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Trmm, a, b, c, false, false, true, 1)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        vector<double> validate = {22, 6, 4};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[i]);
        }

        // Test upper triangle, transpose and A on the left side
        values.clear();
        values = {1, 0, 0, 2, 1, 0, 3, 1, 2};
        for (auto i = 0; i < values.size(); i++) {
            a.SetVal(i, values[i]);
        }
        a.ToMatrix(3, 3);

        DataType output(FLOAT);

        values.clear();
        values = {8, 4, 2};
        for (auto i = 0; i < values.size(); i++) {
            b.SetVal(i, values[i]);
        }
        b.ToMatrix(3, 1);

        SIMPLE_DISPATCH(FLOAT, linear::Trmm, a, b, output, false, true, true, 1)
        REQUIRE(output.GetNCol() == 1);
        REQUIRE(output.GetNRow() == 3);

        validate.clear();
        validate = {8, 20, 32};

        for (auto i = 0; i < output.GetSize(); i++) {
            REQUIRE(output.GetVal(i) == validate[i]);
        }

        // Test lower triangle, not transpose and A on the left side
        values.clear();
        values = {1, 2, 3, 0, 1, 1, 0, 0, 2};
        for (auto i = 0; i < values.size(); i++) {
            a.SetVal(i, values[i]);
        }
        a.ToMatrix(3, 3);

        DataType output2(FLOAT);

        values.clear();
        values = {8, 4, 2};
        for (auto i = 0; i < values.size(); i++) {
            b.SetVal(i, values[i]);
        }
        b.ToMatrix(3, 1);

        SIMPLE_DISPATCH(FLOAT, linear::Trmm, a, b, output2, true, false, true, 1)
        REQUIRE(output2.GetNCol() == 1);
        REQUIRE(output2.GetNRow() == 3);

        validate.clear();
        validate = {8, 20, 32};

        for (auto i = 0; i < output.GetSize(); i++) {
            REQUIRE(output2.GetVal(i) == validate[i]);
        }

        // Test lower triangle, transpose and A on the right side
        values.clear();
        values = {1, 2, 3, 0, 1, 1, 0, 0, 2};
        for (auto i = 0; i < values.size(); i++) {
            a.SetVal(i, values[i]);
        }
        a.ToMatrix(3, 3);

        DataType output3(FLOAT);

        values.clear();
        values = {8, 4, 2};
        for (auto i = 0; i < values.size(); i++) {
            b.SetVal(i, values[i]);
        }
        b.ToMatrix(1, 3);

        SIMPLE_DISPATCH(FLOAT, linear::Trmm, a, b, output3, true, true, false, 1)
        REQUIRE(output3.GetNCol() == 3);
        REQUIRE(output3.GetNRow() == 1);

        validate.clear();
        validate = {8, 20, 32};

        for (auto i = 0; i < output.GetSize(); i++) {
            REQUIRE(output3.GetVal(i) == validate[i]);
        }

    }
}


#ifdef USE_CUDA


/****
 * Solve -> if single
 * QRDecompositionQY
 *
 *
 * TRCON/GECON -> reciprocal condition
 * GETRI -> Solve with one input
 */

void
TEST_GPU() {
    SECTION("Testing Cholesky CUDA Decomposition") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        cout << "Testing CUDA Cholesky Decomposition ..." << endl;
        vector<double> values = {4, 12, -16, 12, 37, -43, -16, -43, 98};
        DataType a(values, DOUBLE);
        a.ToMatrix(3, 3);

        DataType b(DOUBLE);
        SIMPLE_DISPATCH(DOUBLE, linear::Cholesky, a, b)

        vector<double> values_validate = {2, 0, 0, 6, 1, 0, -8, 5,
                                          3}; //Fill Triangle is still not implemented as a CUDA Kernel.
        // Lower Triangle should be set to zeros

        REQUIRE(b.GetNCol() == 3);
        REQUIRE(b.GetNRow() == 3);
        REQUIRE(b.GetPrecision() == DOUBLE);

        for (auto i = 0; i < b.GetSize(); i++) {
            REQUIRE(b.GetVal(i) == values_validate[i]);
        }

        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                CPU);

    }SECTION("Testing CholeskyInv CUDA Decomposition") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        cout << "Testing CUDA Cholesky Inverse ..." << endl;

        vector<double> values = {1, 0, 0, 1, 1, 0, 1, 2, 1.414214};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        DataType b(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::CholeskyInv, a, b, a.GetNCol())
        vector<float> values_validate = {2.5, -2.0, 0.5, -2, 3, -1, 0.5, -1.0,
                                         0.5};
        REQUIRE(b.GetNCol() == 3);
        REQUIRE(b.GetNRow() == 3);


        float error = 0.001;
        for (auto i = 0; i < b.GetSize(); i++) {
            float val =
                    fabs((float) b.GetVal(i) - (float) values_validate[i]) /
                    (float) values_validate[i];
            REQUIRE(val <= error);
        }

        SIMPLE_DISPATCH(FLOAT, linear::CholeskyInv, a, b, 2)
        values_validate = {2, -1, -1, 1};

        REQUIRE(b.GetNCol() == 2);
        REQUIRE(b.GetNRow() == 2);


        for (auto i = 0; i < b.GetSize(); i++) {
            auto val =
                    fabs((float) b.GetVal(i) - (float) values_validate[i]) /
                    (float) values_validate[i];
            REQUIRE(val <= error);
        }
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                CPU);

    }SECTION("Test CUDA Eigen") {

        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        cout << "Testing CUDA Eigen ..." << endl;

        vector<double> values = {1, -1, -1, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(2, 2);


        DataType vals(FLOAT);
        DataType vec(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Eigen, a, vals, &vec)

        REQUIRE(vals.GetSize() == 2);
        REQUIRE(vals.GetVal(0) == 2);


        REQUIRE(vec.GetSize() == 4);
        REQUIRE(vec.GetNCol() == 2);
        REQUIRE(vec.GetNRow() == 2);

        vector<float> validate_values = {-0.7071068, 0.7071068, -0.7071068,
                                         -0.7071068};
        auto err = 0.001;
        for (auto i = 0; i < vec.GetSize(); i++) {
            auto val =
                    fabs((float) vec.GetVal(i) - (float) validate_values[i]) /
                    (float) validate_values[i];
            REQUIRE(val <= err);
        }

    }SECTION("CUDA SVD") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        vector<double> values = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 1, 1, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(9, 4);

        vector<float> validate_values = {3.464102e+00, 1.732051e+00,
                                         1.732051e+00, 1.922963e-16};

        DataType d(FLOAT);
        DataType u(FLOAT);
        DataType v(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::SVD, a, d, u, v, a.GetNCol(),
                        a.GetNCol())
        REQUIRE(d.GetSize() == 4);
        auto err = 0.001;

        DataType dd(9, 4, FLOAT);


        for (auto i = 0; i < dd.GetSize(); i++) {
            dd.SetVal(i, 0);
        }

        for (auto i = 0; i < 4; i++) {
            dd.SetValMatrix(i, i, d.GetVal(i));
        }

        vector<double> temp_vals(81, 0);
        DataType uu(temp_vals, FLOAT);
        uu.ToMatrix(9, 9);

        for (auto i = 0; i < u.GetSize(); i++) {
            uu.SetVal(i, u.GetVal(i));
        }

        DataType vv = v;
        vv.GetData(CPU);
        vv.Transpose();


        DataType temp_one(FLOAT);
        DataType temp_two(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::CrossProduct, uu, dd, temp_one, false,
                        false)

        SIMPLE_DISPATCH(FLOAT, linear::CrossProduct, temp_one, vv, temp_two,
                        false,
                        false)

        DataType temp_three(FLOAT);
        SIMPLE_DISPATCH(FLOAT, math::Round, temp_two, temp_three, 1);

        SIMPLE_DISPATCH(FLOAT, math::PerformRoundOperation, temp_three,
                        temp_two, "abs");

        for (auto i = 0; i < a.GetSize(); i++) {
            REQUIRE(temp_two.GetVal(i) == a.GetVal(i));
        }


    }SECTION("CUDA Gemm & Syrk") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        vector<double> values = {3.12393, -1.16854, -0.304408, -2.15901,
                                 -1.16854, 1.86968, 1.04094, 1.35925,
                                 -0.304408, 1.04094, 4.43374, 1.21072,
                                 -2.15901, 1.35925, 1.21072, 5.57265};

        DataType a(values, DOUBLE);
        a.ToMatrix(4, 4);

        DataType b(values, DOUBLE);
        b.ToMatrix(4, 4);
        DataType output(DOUBLE);

        vector<double> validate_vals = {15.878412787064, -9.08673783542,
                                        -6.13095182416, -20.73289403456,
                                        -9.08673783542, 7.7923056801,
                                        8.56286609912, 13.8991634747,
                                        -6.13095182416, 8.56286609912,
                                        22.300113620064, 14.18705411188,
                                        -20.73289403456, 13.8991634747,
                                        14.18705411188, 39.0291556835};

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output, false,
                        false)
        REQUIRE(output.GetNRow() == 4);
        REQUIRE(output.GetNCol() == 4);

        auto error = 0.001;
        for (auto i = 0; i < validate_vals.size(); i++) {
            auto val =
                    fabs((float) output.GetVal(i) - (float) validate_vals[i]) /
                    (float) validate_vals[i];
            REQUIRE(val <= error);
        }

        values.clear();
        values = {5, 11, 143, 10, 123, 132};
        a.ClearUp();
        a.Allocate(values);
        a.ToMatrix(3, 2);


        values.clear();
        values = {2, 3, 5, 6, 8, 11, 13, 14, 20, 30};
        b.ClearUp();
        b.Allocate(values);
        b.ToMatrix(5, 2);

        output.ClearUp();

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output, false,
                        true)

        REQUIRE(output.GetNRow() == 3);
        REQUIRE(output.GetNCol() == 5);

        validate_vals.clear();
        validate_vals = {120, 1375, 1738, 145, 1632,
                         2145, 165, 1777, 2563, 230,
                         2525, 3498, 340, 3778, 5104};

        for (auto i = 0; i < validate_vals.size(); i++) {
            auto val =
                    fabs((float) output.GetVal(i) - (float) validate_vals[i]) /
                    (float) validate_vals[i];
            REQUIRE(val <= error);
        }

        values.clear();
        values = {2, 3, 5, 6, 8, 11, 13, 14, 20, 30};

        DataType c(values, FLOAT);
        c.ToMatrix(5, 2);

        DataType d(0, FLOAT);

        output.ClearUp();
        output.ConvertPrecision(FLOAT);

        /** SYRK Call **/
        SIMPLE_DISPATCH(FLOAT, linear::CrossProduct, c, d, output, false,
                        true)

        validate_vals.clear();
        validate_vals = {125, 149, 164, 232, 346, 149, 178, 197, 278, 414, 164,
                         197, 221, 310, 460, 232, 278, 310, 436, 648, 346, 414,
                         460, 648, 964};

        for (auto i = 0; i < output.GetSize(); i++) {
            REQUIRE(output.GetVal(i) == validate_vals[i]);
        }

    }SECTION("CUDA backsolve Trsm") {
        cout << "Testing Back Solve CUDA ..." << endl;
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        vector<double> values = {1, 0, 0, 2, 1, 0, 3, 1, 2};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        values.clear();
        values = {8, 4, 2};
        DataType b(values, FLOAT);
        b.ToMatrix(3, 1);
        DataType c(FLOAT);


        SIMPLE_DISPATCH(FLOAT, linear::BackSolve, a, b, c, a.GetNCol(), true,
                        false)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        vector<double> validate = {-1, 3, 1};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[i]);
        }

        c.ClearUp();
        SIMPLE_DISPATCH(FLOAT, linear::BackSolve, a, b, c, a.GetNCol(), true,
                        true)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        validate.clear();
        validate = {8, -12, -5};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[i]);
        }
    }SECTION("CUDA Forward Solve Trsm") {
        cout << "Testing Forward Solve CUDA ..." << endl;
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        vector<double> values = {1, 0, 0, 2, 1, 0, 3, 1, 2};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        values.clear();
        values = {8, 4, 2};
        DataType b(values, FLOAT);
        b.ToMatrix(3, 1);
        DataType c(FLOAT);


        SIMPLE_DISPATCH(FLOAT, linear::BackSolve, a, b, c, a.GetNCol(), false,
                        false)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        vector<double> validate = {8, 4, 1};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[i]);
        }

        c.ClearUp();
        SIMPLE_DISPATCH(FLOAT, linear::BackSolve, a, b, c, a.GetNCol(), false,
                        true)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        validate.clear();
        validate = {8, 4, 1};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[i]);
        }
    }SECTION("CUDA QR Decomposition") {

        auto context = mpcr::kernels::ContextManager::GetOperationContext();
        context->SetOperationPlacement(GPU);

        cout << "Testing CUDA QR Decomposition ..." << endl;
        vector<double> values = {1, 2, 3, 2, 4, 6, 3, 3, 3};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);


        DataType qraux(FLOAT);
        DataType pivot(FLOAT);
        DataType qr(FLOAT);
        DataType rank(FLOAT);


        SIMPLE_DISPATCH(FLOAT, linear::QRDecomposition, a, qr, qraux, pivot,
                        rank)

        REQUIRE(qr.IsMatrix());
        REQUIRE(qr.GetNCol() == 3);
        REQUIRE(qr.GetNRow() == 3);

        DataType Q(FLOAT);
        DataType R(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::QRDecompositionQ, qr, qraux, Q, true)
        SIMPLE_DISPATCH(FLOAT, linear::QRDecompositionR, qr, R, true)


//        REQUIRE(rank.GetVal(0) == 2);

        for (auto i = 0; i < pivot.GetSize(); i++) {
            REQUIRE(pivot.GetVal(i) == 0);
        }

        auto err = 0.001;

        DataType a_reconstruct(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::CrossProduct, Q, R, a_reconstruct, false,
                        false)

        REQUIRE(a_reconstruct.GetNRow() == 3);
        REQUIRE(a_reconstruct.GetNCol() == 3);

        for (auto i = 0; i < a_reconstruct.GetSize(); i++) {
            auto val =
                    fabs((float) a_reconstruct.GetVal(i) - (float) a.GetVal(i)) /
                    (float) a.GetVal(i);
            REQUIRE(val <= err);
        }


    }SECTION("Testing CUDA Solve Two Input") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        cout << "Testing CUDA Solve Two Input ..." << endl;
        vector<double> values = {3, 1, 4, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(2, 2);

        values.clear();
        values = {10, 4};
        DataType b(values, FLOAT);
        b.ToMatrix(2, 1);

        DataType output(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Solve, a, b, output, false)

        values.clear();
        values = {6, -2};

        REQUIRE(output.GetNCol() == 1);
        REQUIRE(output.GetNRow() == 2);

        float error = 0.001;
        for (auto i = 0; i < output.GetSize(); i++) {
            auto val = fabs((float) output.GetVal(i) - (float) values[i]) /
                       (float) values[i];
            REQUIRE(val <= error);
        }

    }SECTION("Testing CUDA Solve One Input") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        cout << "Testing CUDA Solve One Input ..." << endl;
        vector<double> values = {3, 1, 4, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(2, 2);

        DataType b(FLOAT);

        DataType output(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Solve, a, b, output, true)


        vector<double> validate_vals = {-1, 1, 4, -3};


        REQUIRE(output.GetNCol() == 2);
        REQUIRE(output.GetNRow() == 2);

        float error = 0.001;
        for (auto i = 0; i < validate_vals.size(); i++) {
            auto val =
                    fabs((float) output.GetVal(i) - (float) validate_vals[i]) /
                    (float) validate_vals[i];
            REQUIRE(val <= error);
        }

    }SECTION("Testing CUDA R Cond") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        cout << "Testing CUDA R Cond ..." << endl;

        vector<double> values = {100, 2, 3, 3, 2, 1, 300, 3, 3, 400, 5, 6, 4,
                                 44, 56, 1223};
        DataType a(values, FLOAT);
        a.ToMatrix(4, 4);

        double b = 0;


        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "I", false)
        auto val = fabs(b - 0.079608) / 0.079608;
        REQUIRE(val <= 0.001);

        b = 0;

        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "O", false)
        val = fabs(b - 0.074096) / 0.074096;
        REQUIRE(val <= 0.001);

        b = 0;

        DataType temp(4, 4, FLOAT);

        for (auto i = 0; i < a.GetNRow(); i++) {
            for (auto j = 0; j < a.GetNCol(); j++) {
                if (i > j) {
                    temp.SetValMatrix(i, j, a.GetValMatrix(i, j));
                }
                if (i == j) {
                    temp.SetValMatrix(i, j, 1);
                }
            }
        }

        double c = 0;

        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, temp, b, "I", true)

        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                CPU);
        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, temp, c, "I", true)


        if (b != c) {
            val = fabs(b - c) / c;
        } else {
            val = 0;
        }
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        REQUIRE(val <= 0.001);

        b = 0;
        c = 0;

        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, temp, b, "O", true)
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                CPU);
        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, temp, c, "O", true)
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        if (b != c) {
            val = fabs(b - c) / c;
        } else {
            val = 0;
        }

        REQUIRE(val <= 0.001);


    }SECTION("Testing CUDA IsSymmetric") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        cout << "Testing CUDA Matrix Is Symmetric ..." << endl;

        vector<double> values = {2, 3, 6, 3, 4, 5, 6, 5, 9};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        a.GetData(GPU);
        a.FreeMemory(CPU);

        auto isSymmetric = false;
        SIMPLE_DISPATCH(FLOAT, linear::IsSymmetric, a, isSymmetric)
        REQUIRE(isSymmetric == true);

        values.clear();
        values = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        for (auto i = 0; i < values.size(); i++) {
            a.SetVal(i, values[i]);
        }

        a.GetData(GPU);
        a.FreeMemory(CPU);

        isSymmetric = true;
        SIMPLE_DISPATCH(FLOAT, linear::IsSymmetric, a, isSymmetric)
        REQUIRE(isSymmetric == false);

    }SECTION("Testing Trmm") {
        cout << "Testing CUDA Trmm..." << endl;
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                CPU);

        // Test upper triangle, not transpose and A on the left side
        vector<double> values = {1, 0, 0, 2, 1, 0, 3, 1, 2};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        values.clear();
        values = {8, 4, 2};
        DataType b(values, FLOAT);
        b.ToMatrix(3, 1);
        DataType c(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Trmm, a, b, c, false, false, true, 1)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        vector<double> validate = {22, 6, 4};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[i]);
        }

        // Test upper triangle, transpose and A on the left side
        values.clear();
        values = {1, 0, 0, 2, 1, 0, 3, 1, 2};
        for (auto i = 0; i < values.size(); i++) {
            a.SetVal(i, values[i]);
        }
        a.ToMatrix(3, 3);

        DataType output(FLOAT);

        values.clear();
        values = {8, 4, 2};
        for (auto i = 0; i < values.size(); i++) {
            b.SetVal(i, values[i]);
        }
        b.ToMatrix(3, 1);

        SIMPLE_DISPATCH(FLOAT, linear::Trmm, a, b, output, false, true, true, 1)
        REQUIRE(output.GetNCol() == 1);
        REQUIRE(output.GetNRow() == 3);

        validate.clear();
        validate = {8, 20, 32};

        for (auto i = 0; i < output.GetSize(); i++) {
            REQUIRE(output.GetVal(i) == validate[i]);
        }

        // Test lower triangle, not transpose and A on the left side
        values.clear();
        values = {1, 2, 3, 0, 1, 1, 0, 0, 2};
        for (auto i = 0; i < values.size(); i++) {
            a.SetVal(i, values[i]);
        }
        a.ToMatrix(3, 3);

        DataType output2(FLOAT);

        values.clear();
        values = {8, 4, 2};
        for (auto i = 0; i < values.size(); i++) {
            b.SetVal(i, values[i]);
        }
        b.ToMatrix(3, 1);

        SIMPLE_DISPATCH(FLOAT, linear::Trmm, a, b, output2, true, false, true, 1)
        REQUIRE(output2.GetNCol() == 1);
        REQUIRE(output2.GetNRow() == 3);

        validate.clear();
        validate = {8, 20, 32};

        for (auto i = 0; i < output.GetSize(); i++) {
            REQUIRE(output2.GetVal(i) == validate[i]);
        }

        // Test lower triangle, transpose and A on the right side
        values.clear();
        values = {1, 2, 3, 0, 1, 1, 0, 0, 2};
        for (auto i = 0; i < values.size(); i++) {
            a.SetVal(i, values[i]);
        }
        a.ToMatrix(3, 3);

        DataType output3(FLOAT);

        values.clear();
        values = {8, 4, 2};
        for (auto i = 0; i < values.size(); i++) {
            b.SetVal(i, values[i]);
        }
        b.ToMatrix(1, 3);

        SIMPLE_DISPATCH(FLOAT, linear::Trmm, a, b, output3, true, true, false, 1)
        REQUIRE(output3.GetNCol() == 3);
        REQUIRE(output3.GetNRow() == 1);

        validate.clear();
        validate = {8, 20, 32};

        for (auto i = 0; i < output.GetSize(); i++) {
            REQUIRE(output3.GetVal(i) == validate[i]);
        }

    }

}


#endif


#ifdef USING_HALF


void
TEST_HALF_GEMM() {
    SECTION("GEMM HALF") {
        cout << "Testing CUDA HALF Matrix ..." << endl;
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        vector<double> values = {3.12393, -1.16854, -0.304408, -2.15901,
                                 -1.16854, 1.86968, 1.04094, 1.35925,
                                 -0.304408, 1.04094, 4.43374, 1.21072,
                                 -2.15901, 1.35925, 1.21072, 5.57265};


        DataType a(values, HALF, GPU);
        a.ToMatrix(4, 4);

        DataType b(values, HALF, GPU);
        b.ToMatrix(4, 4);
        DataType output(HALF, GPU);

        vector<double> validate_vals = {15.878412787064, -9.08673783542,
                                        -6.13095182416, -20.73289403456,
                                        -9.08673783542, 7.7923056801,
                                        8.56286609912, 13.8991634747,
                                        -6.13095182416, 8.56286609912,
                                        22.300113620064, 14.18705411188,
                                        -20.73289403456, 13.8991634747,
                                        14.18705411188, 39.0291556835};

        SIMPLE_DISPATCH_WITH_HALF(HALF, linear::CrossProduct, a, b, output,
                                  false,
                                  false)
        REQUIRE(output.GetPrecision() == HALF);
        REQUIRE(output.GetNRow() == 4);
        REQUIRE(output.GetNCol() == 4);


        auto error = 0.001;
        for (auto i = 0; i < validate_vals.size(); i++) {
            auto val =
                    fabs((float) output.GetVal(i) - (float) validate_vals[i]) /
                    (float) validate_vals[i];
            REQUIRE(val <= error);
        }

    }SECTION("Testing Half Syrk using gemm") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);
        int size = 16;
        std::vector<double> values;
        // Random number generator
        std::random_device rd; // Seed for the random number engine
        std::mt19937 rng(rd()); // Mersenne-Twister random number engine
        std::uniform_real_distribution<> dist(0.0,
                                              1.0); // Uniform distribution between 0 and 1

        // Generate and store the numbers
        for (int i = 0; i < size; i++) {
            double num = dist(rng);
            num = std::round(num * 100) / 100.0; // Round to two decimal places
            values.push_back(num);
        }

        DataType a(values, HALF, GPU);
        a.SetDimensions(4, 4);

        DataType c(HALF, GPU);
        DataType b(HALF, GPU);
        DataType validate(FLOAT, GPU);


        a.ConvertPrecision(HALF);

        SIMPLE_DISPATCH_WITH_HALF(HALF, linear::CrossProduct, a, b, c,
                                  false,
                                  false)

        a.ConvertPrecision(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::CrossProduct, a, b, validate,
                        false, false)

        auto error = 0.001;
        for (auto i = 0; i < size; i++) {
            auto val =
                    fabs((float) c.GetVal(i) - (float) validate.GetVal(i)) /
                    (float) validate.GetVal(i);
            REQUIRE(val <= error);
        }

    }
}

#endif


TEST_CASE("LinearAlgebra", "[Linear Algebra]") {
    mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
            CPU);
    TEST_LINEAR_ALGEBRA();

#ifdef USE_CUDA
    TEST_GPU();
#endif

#ifdef USING_HALF
    //TEST_HALF_GEMM();
#endif

}
