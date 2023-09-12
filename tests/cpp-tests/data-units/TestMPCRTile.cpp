/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <libraries/catch/catch.hpp>
#include <data-units/MPCRTile.hpp>
#include <utilities/MPCRDispatcher.hpp>


using namespace mpcr::precision;
using namespace std;


void
TEST_MPR_TILE() {
    SECTION("MPR Tile Initialization") {
        cout << "Testing MPR Tile" << endl;
        cout << "Testing MPR Tile Construction ..." << endl;

        vector <double> data;
        auto size_data = 24;
        data.resize(size_data);
        for (auto i = 0; i < size_data; i++) {
            data[ i ] = i + 1;
        }

        vector <string> precisions;
        auto size_precision = 4;
        precisions.resize(size_precision);
        vector <string> comp_precision = {"float", "double"};
        for (auto i = 0; i < size_precision; i++) {
            precisions[ i ] = comp_precision[ random() %
                                              comp_precision.size() ];
        }


        MPCRTile a(6, 4, 3, 2, data, precisions);

        REQUIRE(a.GetNRow() == 6);
        REQUIRE(a.GetNCol() == 4);
        REQUIRE(a.GetTileNRow() == 3);
        REQUIRE(a.GetTileNCol() == 2);
        REQUIRE(a.GetTileSize() == 3 * 2);
        REQUIRE(a.GetMatrixSize() == 6 * 4);

        vector <vector <double>> validate_vals;
        validate_vals.resize(4);
        validate_vals[ 0 ] = {1, 2, 3, 7, 8, 9};
        validate_vals[ 1 ] = {4, 5, 6, 10, 11, 12};
        validate_vals[ 2 ] = {13, 14, 15, 19, 20, 21};
        validate_vals[ 3 ] = {16, 17, 18, 22, 23, 24};

        auto j = 0;
        auto tiles = a.GetTiles();
        for (auto &tile: tiles) {
            auto precision_temp = GetInputPrecision(precisions[ j ]);
            REQUIRE(precision_temp == tile->GetPrecision());
            for (auto i = 0; i < a.GetTileSize(); i++) {
                REQUIRE(validate_vals[ j ][ i ] == tile->GetVal(i));
            }
            j++;
        }

        a.SetVal(3, 3, 0);
        REQUIRE(tiles[ 3 ]->GetValMatrix(0, 1) == 0);

        a.SetVal(0, 2, 1);
        REQUIRE(tiles[ 2 ]->GetValMatrix(0, 0) == 1);

        a.SetVal(1, 2, 2);
        REQUIRE(tiles[ 2 ]->GetValMatrix(1, 0) == 2);

        a.SetVal(0, 0, -3);
        REQUIRE(tiles[ 0 ]->GetValMatrix(0, 0) == -3);

        a.SetVal(5, 1, 17);
        REQUIRE(tiles[ 1 ]->GetValMatrix(2, 1) == 17);

        REQUIRE(a.GetVal(3, 3) == 0);
        REQUIRE(a.GetVal(0, 2) == 1);
        REQUIRE(a.GetVal(1, 2) == 2);
        REQUIRE(a.GetVal(0, 0) == -3);
        REQUIRE(a.GetVal(5, 1) == 17);

        for (auto &x: tiles) {
            x->ConvertPrecision(FLOAT);
        }

        tiles = a.GetTiles();
        for (auto &tile: tiles) {
            REQUIRE(tile->GetPrecision() == FLOAT);
        }

        a.ChangePrecision(0, 0, DOUBLE);
        REQUIRE(tiles[ 0 ]->GetPrecision() == DOUBLE);

        a.ChangePrecision(1, 0, DOUBLE);
        REQUIRE(tiles[ 1 ]->GetPrecision() == DOUBLE);

        a.ChangePrecision(0, 1, DOUBLE);
        REQUIRE(tiles[ 2 ]->GetPrecision() == DOUBLE);

        a.ChangePrecision(1, 1, DOUBLE);
        REQUIRE(tiles[ 3 ]->GetPrecision() == DOUBLE);

        REQUIRE(a.IsMPCRTile() == true);


    } SECTION("Get and Insert Tile") {
        cout << "Testing MPR Tile Getters/Setter for Tiles ..." << endl;
        vector <double> data;
        auto size_data = 24;
        data.resize(size_data);
        for (auto i = 0; i < size_data; i++) {
            data[ i ] = i + 1;
        }

        vector <string> precisions;
        auto size_precision = 4;
        precisions.resize(size_precision);
        vector <string> comp_precision = {"float", "double"};
        for (auto i = 0; i < size_precision; i++) {
            precisions[ i ] = comp_precision[ random() %
                                              comp_precision.size() ];
        }


        MPCRTile a(6, 4, 3, 2, data, precisions);


        a.InsertTile(nullptr, 0, 0);
        auto tile_temp = a.GetTile(0, 0);
        REQUIRE(tile_temp == nullptr);

        auto temp_tile_test = new DataType(9, FLOAT);
        a.InsertTile(temp_tile_test, 1, 1);

        temp_tile_test = a.GetTile(1, 1);
        REQUIRE(temp_tile_test->GetPrecision() == FLOAT);
        REQUIRE(temp_tile_test->GetSize() == 9);
        REQUIRE(temp_tile_test->IsMatrix() == false);

        for (auto i = 0; i < temp_tile_test->GetSize(); i++) {
            REQUIRE(temp_tile_test->GetVal(i) == 1.5);
        }


    }SECTION("Tile Getter and Setter Indexing") {

        vector <double> data;
        auto size_data = 120;
        data.resize(size_data);
        for (auto i = 0; i < size_data; i++) {
            data[ i ] = i + 1;
        }

        vector <string> precisions;
        auto precision_size = 15;

        precisions.resize(precision_size);
        for (auto i = 0; i < precision_size; i++) {
            precisions[ i ] = "half";
        }
        MPCRTile a(10, 12, 2, 4, data, precisions);

        a.ChangePrecision(4, 2, FLOAT);
        auto tile_temp = a.GetTile(4, 2);
        REQUIRE(tile_temp->GetPrecision() == FLOAT);
        REQUIRE(tile_temp->GetValMatrix(0, 0) == 89);

        DataType *copy_tile = new DataType(*tile_temp);

        a.InsertTile(copy_tile, 2, 1);
        REQUIRE(a.GetTile(2, 1)->GetPrecision() == FLOAT);
        REQUIRE(a.GetVal(4, 4) == 89);

    }SECTION("Test Fill with Zeros") {
        cout << "Testing Fill With Zeros ..." << endl;

        MPCRTile a(6, 6, 3, 2);
        a.FillWithZeros();

        auto nrows = a.GetNRow();
        auto ncols = a.GetNCol();

        REQUIRE(nrows == 6);
        REQUIRE(ncols == 6);

        for (auto i = 0; i < nrows; i++) {
            for (auto j = 0; j < ncols; j++) {
                REQUIRE(a.GetVal(i, j) == 0);
            }
        }
    }SECTION("Test Sum and Product") {
        vector <double> values;
        vector <string> precisions(5, "float");
        auto size = 20;
        values.resize(size);
        double validate_prod = 1;
        auto validate_sum = 0;

        for (auto i = 0; i < size; i++) {
            values[ i ] = i + 1;
            validate_sum += values[ i ];
            validate_prod *= values[ i ];
        }

        MPCRTile a(2, 10, 2, 2, values, precisions);
        auto sum = a.Sum();
        REQUIRE(sum == validate_sum);
        auto prod = a.Product();
        REQUIRE(prod == validate_prod);
    }


}


TEST_CASE("MPCRTile Test", "[MPCRTile]") {
    TEST_MPR_TILE();
}
