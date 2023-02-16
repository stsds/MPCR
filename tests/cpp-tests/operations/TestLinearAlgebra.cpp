


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
        DataType a(values, DOUBLE);
        a.ToMatrix(9, 4);
        a.Print();


        vector<float>validate_values = {3.464102e+00, 1.732051e+00, 1.732051e+00, 1.922963e-16};

        DataType d(DOUBLE);
        DataType u(DOUBLE);
        DataType v(DOUBLE);

        SIMPLE_DISPATCH(DOUBLE, linear::SVD, a,d,u,v,a.GetNCol(),a.GetNCol())
        REQUIRE(d.GetSize()==4);
        auto err=0.001;
//        for(auto i=0;i<v.GetSize();i++){
////            float val= fabs((float)d.GetVal(i)-(float)values[i])/(float)values[i];
////            REQUIRE(val<err);
//            cout<<v.GetVal(i)<<endl;
//        }
d.Print();
        cout<<"-------------------"<<endl;
        u.Print();
        cout<<"-------------------"<<endl;
            v.Print();

//
//        cout<<"---------------------"<<endl;
//        for(auto i=0;i<validate_values.size();i++){
//            cout<<validate_values[i]<<endl;
//        }


    }
}


TEST_CASE("LinearAlgebra", "[Linear Algebra]") {
    TEST_LINEAR_ALGEBRA();
}
