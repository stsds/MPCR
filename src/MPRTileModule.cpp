
#include <data-units/MPRTile.hpp>
#include <operations/TileLinearAlgebra.hpp>

/** Expose C++ class to R to be able to use Wrap and As
 *  Allows C++ to Send and Receive Class object from R
 **/
RCPP_EXPOSED_CLASS(MPRTile)

/** Expose C++ Object With the Given functions **/
RCPP_MODULE(MPRTile) {


    void (MPRTile::*pChangePrecision)(const size_t &, const size_t &,
                                      const std::string &) =&MPRTile::ChangePrecision;

    using namespace Rcpp;
    class_ <MPRTile>("MPRTile")
        .constructor <size_t, size_t, size_t, size_t,
            std::vector <double>, std::vector <std::string> >()
        .property("Row", &MPRTile::GetNRow)
        .property("Col", &MPRTile::GetNCol)
        .property("Size", &MPRTile::GetMatrixSize)
        .property("TileRow", &MPRTile::GetTileNRow)
        .property("TileCol", &MPRTile::GetTileNCol)
        .property("TileSize", &MPRTile::GetTileSize)
        .method("PrintTile", &MPRTile::PrintTile)
        .method("ChangeTilePrecision", pChangePrecision)
        .method("MPRTile.SetVal", &MPRTile::SetVal)
        .method("MPRTile.GetVal", &MPRTile::GetVal)
        .method("show", &MPRTile::GetType)
        .method("MPRTile.print", &MPRTile::Print);


    function("MPR.gemm", &mpr::operations::linear::TileGemm,
             List::create(_[ "a" ], _[ "b" ], _[ "c" ], _[ "alpha" ] = 1,
                          _[ "beta" ] = 0));

    function("MPRTile.chol", &mpr::operations::linear::TileCholesky,
             List::create(_[ "x" ], _[ "overwrite_input" ] = true));

}
