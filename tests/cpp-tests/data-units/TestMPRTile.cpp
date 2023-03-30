

#include <libraries/catch/catch.hpp>
#include <data-units/MPRTile.hpp>
#include <utilities/MPRDispatcher.hpp>


using namespace mpr::precision;
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


        MPRTile a(6, 4, 3, 2, data, precisions);
        a.Print();
        REQUIRE(a.GetNRow() == 6);
        REQUIRE(a.GetNCol() == 4);
        REQUIRE(a.GetTileNRow() == 3);
        REQUIRE(a.GetTileNCol() == 2);
        REQUIRE(a.GetTileSize()==3*2);
        REQUIRE(a.GetMatrixSize()==6*4);

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

        a.SetVal(3,3,0);
        REQUIRE(tiles[3]->GetValMatrix(0,1)==0);

        a.SetVal(0,2,1);
        REQUIRE(tiles[2]->GetValMatrix(0,0)==1);

        a.SetVal(1,2,2);
        REQUIRE(tiles[2]->GetValMatrix(1,0)==2);

        a.SetVal(0,0,-3);
        REQUIRE(tiles[0]->GetValMatrix(0,0)==-3);

        a.SetVal(5,1,17);
        REQUIRE(tiles[1]->GetValMatrix(2,1)==17);

        REQUIRE(a.GetVal(3,3)==0);
        REQUIRE(a.GetVal(0,2)==1);
        REQUIRE(a.GetVal(1,2)==2);
        REQUIRE(a.GetVal(0,0)==-3);
        REQUIRE(a.GetVal(5,1)==17);

        for(auto &x:tiles){
            x->ConvertPrecision(FLOAT);
        }

        tiles=a.GetTiles();
        for (auto &tile: tiles) {
            REQUIRE( tile->GetPrecision()==FLOAT);
        }

        a.ChangePrecision(0,0,DOUBLE);
        REQUIRE(tiles[0]->GetPrecision()==DOUBLE);

        a.ChangePrecision(1,0,DOUBLE);
        REQUIRE(tiles[1]->GetPrecision()==DOUBLE);

        a.ChangePrecision(0,1,DOUBLE);
        REQUIRE(tiles[2]->GetPrecision()==DOUBLE);

        a.ChangePrecision(1,1,DOUBLE);
        REQUIRE(tiles[3]->GetPrecision()==DOUBLE);

        REQUIRE(a.IsMPRTile()==true);


    }

}


TEST_CASE("MPRTile Test", "[MPRTile]") {
    TEST_MPR_TILE();
}
