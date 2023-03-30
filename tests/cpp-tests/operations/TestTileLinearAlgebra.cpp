
#include <libraries/catch/catch.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <operations/TileLinearAlgebra.hpp>
#include <operations/LinearAlgebra.hpp>


using namespace std;


void
TEST_TILE_LINEAR_ALGEBRA() {
    SECTION("Tile Gemm") {
        cout << "--------------------------------------" << endl;
        cout << "Testing Tile Matrix Multiplication..." << endl;

        vector <double> values = {3.12393, -1.16854, -0.304408, -2.15901,
                                  -1.16854, 1.86968, 1.04094, 1.35925,
                                  -0.304408, 1.04094, 4.43374, 1.21072,
                                  -2.15901, 1.35925, 1.21072, 5.57265};
        vector <string> precision_a = {"float", "double", "float", "float"};
        vector <string> precision_b = {"float", "float", "double", "float"};


        MPRTile a(4, 4, 2, 2, values, precision_a);
        MPRTile b(4, 4, 2, 2, values, precision_b);
        auto counter = 0;

        for (auto i = 0; i < a.GetNCol(); i++) {
            for (auto j = 0; j < a.GetNRow(); j++) {
                a.SetVal(j, i, values[ counter ]);
                b.SetVal(j, i, values[ counter ]);
                counter++;
            }
        }


        auto pMatrix_c = mpr::operations::linear::TileGemm(a, b);

        REQUIRE(pMatrix_c->GetNRow() == 4);
        REQUIRE(pMatrix_c->GetNCol() == 4);
        REQUIRE(pMatrix_c->GetTileNCol() == 2);
        REQUIRE(pMatrix_c->GetTileNRow() == 2);
        REQUIRE(pMatrix_c->GetTilePerCol() == 2);
        REQUIRE(pMatrix_c->GetTilePerRow() == 2);


        auto error = 0.01;


        vector <double> validate_vals = {15.878412787064, -9.08673783542,
                                         -6.13095182416, -20.73289403456,
                                         -9.08673783542, 7.7923056801,
                                         8.56286609912, 13.8991634747,
                                         -6.13095182416, 8.56286609912,
                                         22.300113620064, 14.18705411188,
                                         -20.73289403456, 13.8991634747,
                                         14.18705411188, 39.0291556835};


        DataType temp_print(validate_vals, FLOAT);
        temp_print.SetDimensions(4, 4);

        counter = 0;
        for (auto i = 0; i < pMatrix_c->GetNCol(); i++) {
            for (auto j = 0; j < pMatrix_c->GetNRow(); j++) {
                auto val = pMatrix_c->GetVal(j, i);
                auto temp_error_perc = fabs(val - validate_vals[ counter ]) /
                                       validate_vals[ counter ];
                REQUIRE(temp_error_perc <= error);
                counter++;

            }
        }

        delete pMatrix_c;

    }SECTION("Tile Cholesky decomposition") {

        cout << "Testing Tile Cholesky Decomposition ..." << endl;
        vector <double> values = {9, 0, -27, 18, 0, 9, -9, -27, -27, -9, 99,
                                  -27, 18, -27, -27, 121};

        vector <string> precision_a = {"float", "double", "float", "double"};


        MPRTile a(4, 4, 2, 2, values, precision_a);

        auto counter = 0;

        for (auto i = 0; i < a.GetNCol(); i++) {
            for (auto j = 0; j < a.GetNRow(); j++) {
                a.SetVal(j, i, values[ counter ]);
                counter++;
            }
        }

        DataType temp_tile(values, FLOAT);
        temp_tile.SetDimensions(4, 4);
        DataType output_temp_chol(FLOAT);

        SIMPLE_DISPATCH(FLOAT, mpr::operations::linear::Cholesky, temp_tile,
                        output_temp_chol, false)

        auto pMatrix_output = mpr::operations::linear::TileCholesky(a, false);

        REQUIRE(pMatrix_output->GetNCol() == 4);
        REQUIRE(pMatrix_output->GetNRow() == 4);

        auto error = 0.01;

        for (auto i = 0; i < pMatrix_output->GetNCol(); i++) {
            for (auto j = 0; j < pMatrix_output->GetNRow(); j++) {
                auto val = pMatrix_output->GetVal(j, i);
                auto validate_val = output_temp_chol.GetValMatrix(j, i);
                auto temp_error_perc = 0;
                if (validate_val != 0) {
                    temp_error_perc = fabs(val - validate_val) / validate_val;
                } else {
                    temp_error_perc = fabs(val - validate_val);
                }
                REQUIRE(temp_error_perc <= error);
            }
        }

        delete pMatrix_output;

    }SECTION("Test Tile Cholesky") {
        cout << "Testing Tile Cholesky Decomposition ..." << endl;
        vector <double> values = {1.21, 0.18, 0.13, 0.41, 0.06, 0.23,
                                  0.18, 0.64, 0.10, -0.16, 0.23, 0.07,
                                  0.13, 0.10, 0.36, -0.10, 0.03, 0.18,
                                  0.41, -0.16, -0.10, 1.05, -0.29, -0.08,
                                  0.06, 0.23, 0.03, -0.29, 1.71, -0.10,
                                  0.23, 0.07, 0.18, -0.08, -0.10, 0.36};

        vector <string> precision_a = {"float", "double", "float", "float",
                                       "double", "double", "float", "float",
                                       "double"};


        MPRTile a(6, 6, 2, 2, values, precision_a);
        DataType temp_chol(values, FLOAT);
        temp_chol.SetDimensions(6, 6);

        auto counter = 0;

        for (auto i = 0; i < a.GetNCol(); i++) {
            for (auto j = 0; j < a.GetNRow(); j++) {
                a.SetVal(j, i, values[ counter ]);
                counter++;
            }
        }
        REQUIRE(a.GetTilePerRow() == 3);
        REQUIRE(a.GetTilePerCol() == 3);

        DataType output_temp_chol(FLOAT);
        SIMPLE_DISPATCH(FLOAT, mpr::operations::linear::Cholesky, temp_chol,
                        output_temp_chol, false)

        auto pMatrix_output = mpr::operations::linear::TileCholesky(a);

        REQUIRE(pMatrix_output->GetNCol() == 6);
        REQUIRE(pMatrix_output->GetNRow() == 6);

        auto error = 0.01;

        for (auto i = 0; i < pMatrix_output->GetNCol(); i++) {
            for (auto j = 0; j < pMatrix_output->GetNRow(); j++) {

                auto val = pMatrix_output->GetVal(j, i);
                auto validate_val = output_temp_chol.GetValMatrix(j, i);
                auto temp_error_perc = 0;
                if (validate_val != 0) {
                    temp_error_perc = fabs(val - validate_val) / validate_val;
                } else {
                    temp_error_perc = fabs(val - validate_val);
                }
                REQUIRE(temp_error_perc <= error);
            }
        }

    }
}


TEST_CASE("TileLinearAlgebra", "[Tile Linear Algebra]") {
    TEST_TILE_LINEAR_ALGEBRA();
}
