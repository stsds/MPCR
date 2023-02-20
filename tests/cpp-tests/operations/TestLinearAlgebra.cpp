


#include <libraries/catch/catch.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <operations/LinearAlgebra.hpp>


using namespace std;
using namespace mpr::precision;
using namespace mpr::operations;


void
TEST_LINEAR_ALGEBRA() {
    SECTION("Test CrossProduct") {
        DataType a(5, 10, FLOAT);
        DataType b(10, 5, FLOAT);
        DataType output(FLOAT);

        for (auto i = 0; i < a.GetSize(); i++) {
            a.SetVal(i, i + 1);
        }


        for (auto i = 0; i < b.GetSize(); i++) {
            b.SetVal(i, i + 1);
        }

//        SIMPLE_DISPATCH(FLOAT,linear::CrossProduct,a,b,output)
//        REQUIRE(output.GetNRow()==5);
//        REQUIRE(output.GetNCol()==5);

//        output.Print();
    }SECTION("Test Symmetric") {
        cout << "Testing Matrix Is Symmetric ..." << endl;
        vector <double> values = {2, 3, 6, 3, 4, 5, 6, 5, 9};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);
        auto isSymmetric = false;
        SIMPLE_DISPATCH(FLOAT, linear::IsSymmetric, a, isSymmetric)
        REQUIRE(isSymmetric == true);

        values.clear();
        values = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        for (auto i = 0; i < values.size(); i++) {
            a.SetVal(i, values[ i ]);
        }

        isSymmetric = true;
        SIMPLE_DISPATCH(FLOAT, linear::IsSymmetric, a, isSymmetric)
        REQUIRE(isSymmetric == false);

    }SECTION("Testing Transpose") {
        cout << "Testing Matrix Transpose ..." << endl;
        vector <double> values = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
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
        vector <double> values = {4, 12, -16, 12, 37, -43, -16, -43, 98};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);
        DataType b(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Cholesky, a, b)

        vector <double> values_validate = {2, 0, 0, 6, 1, 0, -8, 5, 3};

        REQUIRE(b.GetNCol() == 3);
        REQUIRE(b.GetNRow() == 3);
        for (auto i = 0; i < b.GetSize(); i++) {
            REQUIRE(b.GetVal(i) == values_validate[ i ]);
        }

    }SECTION("Test Cholesky Inverse ") {
        cout << "Testing Cholesky Inverse ..." << endl;

        vector <double> values = {1, 0, 0, 1, 1, 0, 1, 2, 1.414214};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        DataType b(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::CholeskyInv, a, b, a.GetNCol())
        vector <float> values_validate = {2.5, -2.0, 0.5, -2, 3, -1, 0.5, -1.0,
                                          0.5};
        REQUIRE(b.GetNCol() == 3);
        REQUIRE(b.GetNRow() == 3);

        float error = 0.001;
        for (auto i = 0; i < b.GetSize(); i++) {
            float val =
                fabs((float) b.GetVal(i) - (float) values_validate[ i ]) /
                (float) values_validate[ i ];
            REQUIRE(val <= error);
        }

        SIMPLE_DISPATCH(FLOAT, linear::CholeskyInv, a, b, 2)
        values_validate = {2, -1, -1, 1};

        REQUIRE(b.GetNCol() == 2);
        REQUIRE(b.GetNRow() == 2);


        for (auto i = 0; i < b.GetSize(); i++) {
            auto val =
                fabs((float) b.GetVal(i) - (float) values_validate[ i ]) /
                (float) values_validate[ i ];
            REQUIRE(val <= error);
        }
    }SECTION("Testing Solve") {
        cout << "Testing Solve ..." << endl;
        vector <double> values = {3, 1, 4, 1};
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
            auto val = fabs((float) output.GetVal(i) - (float) values[ i ]) /
                       (float) values[ i ];
            REQUIRE(val <= error);
        }

    }SECTION("Testing Back solve") {
        cout << "Testing Back Solve ..." << endl;

        vector <double> values = {1, 0, 0, 2, 1, 0, 3, 1, 2};
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

        vector <double> validate = {-1, 3, 1};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[ i ]);
        }

        c.ClearUp();
        SIMPLE_DISPATCH(FLOAT, linear::BackSolve, a, b, c, a.GetNCol(), true,
                        true)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        validate.clear();
        validate = {8, -12, -5};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[ i ]);
        }
    }SECTION("Testing Forward Solve") {
        cout << "Testing Forward Solve ..." << endl;

        vector <double> values = {1, 0, 0, 2, 1, 0, 3, 1, 2};
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

        vector <double> validate = {8, 4, 1};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[ i ]);
        }

        c.ClearUp();
        SIMPLE_DISPATCH(FLOAT, linear::BackSolve, a, b, c, a.GetNCol(), false,
                        true)
        REQUIRE(c.GetNCol() == 1);
        REQUIRE(c.GetNRow() == 3);

        validate.clear();
        validate = {8, 4, 1};

        for (auto i = 0; i < c.GetSize(); i++) {
            REQUIRE(c.GetVal(i) == validate[ i ]);
        }

    }SECTION("SVD") {
        cout << "Testing Singular Value Decomposition ..." << endl;

        vector <double> values = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 1, 1, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(9, 4);
        a.Print();


        vector <float> validate_values = {3.464102e+00, 1.732051e+00,
                                          1.732051e+00, 1.922963e-16};

        DataType d(FLOAT);
        DataType u(FLOAT);
        DataType v(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::SVD, a, d, u, v, a.GetNCol(),
                        a.GetNCol())
        REQUIRE(d.GetSize() == 4);
        auto err = 0.001;
////        for(auto i=0;i<v.GetSize();i++){
//////            float val= fabs((float)d.GetVal(i)-(float)values[i])/(float)values[i];
//////            REQUIRE(val<err);
////            cout<<v.GetVal(i)<<endl;
////        }
//        d.Print();
//        cout << "-------------------" << endl;
//        u.Print();
//        cout << "-------------------" << endl;
//        v.Print();

////
////        cout<<"---------------------"<<endl;
////        for(auto i=0;i<validate_values.size();i++){
////            cout<<validate_values[i]<<endl;
////        }
    }SECTION("Eigen") {
        cout << "Testing Eigen ..." << endl;

        vector <double> values = {1, -1, -1, 1};
        DataType a(values, FLOAT);
        a.ToMatrix(2, 2);
//        a.Print();

        DataType vals(FLOAT);
        DataType vec(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Eigen, a, vals, &vec)
//        vals.Print();
//        vec.Print();
        REQUIRE(vals.GetSize() == 2);
        REQUIRE(vals.GetVal(0) == 2);
        REQUIRE(vals.GetVal(1) == 0);

        REQUIRE(vec.GetSize() == 4);
        REQUIRE(vec.GetNCol() == 2);
        REQUIRE(vec.GetNRow() == 2);


        vector <float> validate_values = {-0.7071068, 0.7071068, -0.7071068,
                                          -0.7071068};
        auto err = 0.001;
        for (auto i = 0; i < vec.GetSize(); i++) {
            auto val =
                fabs((float) vec.GetVal(i) - (float) validate_values[ i ]) /
                (float) validate_values[ i ];
            REQUIRE(val <= err);
        }
    }SECTION("Norm") {
        cout << "Testing Norm ..." << endl;

        vector <double> values = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5,
                                  6, 7, 8, 9, 10};
        DataType a(values, FLOAT);
        a.ToMatrix(10, 2);

        DataType norm_val(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::Norm, a, "O", norm_val)
        REQUIRE(norm_val.GetSize() == 1);
        REQUIRE(norm_val.GetVal(0) == 55);

        SIMPLE_DISPATCH(FLOAT, linear::Norm, a, "I", norm_val)
        REQUIRE(norm_val.GetSize() == 1);
        REQUIRE(norm_val.GetVal(0) == 11);

        SIMPLE_DISPATCH(FLOAT, linear::Norm, a, "M", norm_val)
        REQUIRE(norm_val.GetSize() == 1);
        REQUIRE(norm_val.GetVal(0) == 10);

        SIMPLE_DISPATCH(FLOAT, linear::Norm, a, "F", norm_val)
        REQUIRE(norm_val.GetSize() == 1);
        auto val = fabs(norm_val.GetVal(0) - 19.87461) / 19.87461;
        REQUIRE(val <= 0.001);
    }SECTION("QR Decomposition") {
        cout << "Testing QR Decomposition ..." << endl;
        vector <double> values = {1, 2, 3, 2, 4, 6, 3, 3, 3};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        DataType qraux(FLOAT);
        DataType pivot(FLOAT);
        DataType qr(FLOAT);
        size_t rank = 0;

        SIMPLE_DISPATCH(FLOAT, linear::QRDecomposition, a, qr, qraux, pivot,
                        rank)

        vector <float> validate_vals = {-7.48331, 0.42179, 0.63269, -4.81070,
                                        1.96396, 0.85977, -3.7417, 0, 0};


        REQUIRE(qr.IsMatrix());
        REQUIRE(qr.GetNCol() == 3);
        REQUIRE(qr.GetNRow() == 3);

        auto err = 0.001;
        for (auto i = 0; i < qr.GetSize(); i++) {

            if (validate_vals[ i ] != 0) {
                auto val =
                    fabs(qr.GetVal(i) - validate_vals[ i ]) /
                    validate_vals[ i ];
                REQUIRE(val <= err);
            } else {
                REQUIRE(qr.GetVal(i) <= 1e-07);
            }

        }

        REQUIRE(rank == 2);

        validate_vals.clear();
        validate_vals = {1.2673, 1.1500, 0.0000};
        REQUIRE(qraux.GetSize() == 3);

        for (auto i = 0; i < 2; i++) {
            auto val =
                fabs(qraux.GetVal(i) - validate_vals[ i ]) / validate_vals[ i ];
            REQUIRE(val <= err);
        }

        REQUIRE(qraux.GetVal(2) == 0);

        validate_vals.clear();
        validate_vals = {2, 3, 1};
        for (auto i = 0; i < pivot.GetSize(); i++) {
            REQUIRE(pivot.GetVal(i) == validate_vals[ i ]);
        }

    }SECTION("Test QR.R") {

        cout << "Testing QR Auxiliaries ..." << endl;
        cout << "Testing QR.R ..." << endl;


        vector <double> values = {-7.48331, 0.42179, 0.63269, -4.81070, 1.96396,
                                  0.85977, -3.7417, 0, 0};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);
        DataType r(FLOAT);

        SIMPLE_DISPATCH(FLOAT, linear::QRDecompositionR, a, r, true)

        vector <float> validate_vals = {-7.4833, 0, 0, -4.8107, 1.9640, 0,
                                        -3.7417, 0, 0};
        auto error = 0.001;
        for (auto i = 0; i < r.GetSize(); i++) {
            if (validate_vals[ i ] != 0) {
                auto val =
                    fabs(r.GetVal(i) - validate_vals[ i ]) / validate_vals[ i ];
                REQUIRE(val <= error);
            }
        }

    }SECTION("Test QR.Q") {

        cout << "Testing QR.Q ..." << endl;

        vector <double> values = {1, 2, 3, 2, 4, 6, 3, 3, 3};
        DataType a(values, FLOAT);
        a.ToMatrix(3, 3);

        DataType qraux(FLOAT);
        DataType pivot(FLOAT);
        DataType qr(FLOAT);
        size_t rank = 0;

        SIMPLE_DISPATCH(FLOAT, linear::QRDecomposition, a, qr, qraux, pivot,
                        rank)


        DataType qr_q(FLOAT);
        SIMPLE_DISPATCH(FLOAT, linear::QRDecompositionQ, qr, qraux, qr_q, FALSE)

//        qr_q.Print();


    }SECTION("Testing R Cond") {

        cout << "Testing R Cond ..." << endl;

        vector <double> values = {100, 2, 3, 3, 2, 1, 300, 3, 3, 400, 5, 6, 4,
                                  44, 56, 1223};
        DataType a(values, FLOAT);
        a.ToMatrix(4, 4);

        DataType b(FLOAT);


        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "I", false)
        auto val = fabs(b.GetVal(0) - 0.079608) / 0.079608;
        REQUIRE(val <= 0.001);

        b.ClearUp();

        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "O", false)
        val = fabs(b.GetVal(0) - 0.074096) / 0.074096;
        REQUIRE(val <= 0.001);

        b.ClearUp();

        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "I", true)
        val = fabs(b.GetVal(0) - 1.3189e-05) / 1.3189e-05;
        b.Print();
//        REQUIRE(val <= 0.001);

        b.ClearUp();

        SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "O", true)
        b.Print();
        val = fabs(b.GetVal(0) - 1.334e-05) / 1.334e-05;
//        REQUIRE(val <= 0.001);
    }
}


TEST_CASE("LinearAlgebra", "[Linear Algebra]") {
    TEST_LINEAR_ALGEBRA();
}
