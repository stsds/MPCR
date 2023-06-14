
#include <data-units/MPRTile.hpp>
#include <operations/TileLinearAlgebra.hpp>
#include <adapters/RHelpers.hpp>

/** Expose C++ class to R to be able to use Wrap and As
 *  Allows C++ to Send and Receive Class object from R
 **/
RCPP_EXPOSED_CLASS(MPRTile)
RCPP_EXPOSED_CLASS(DataType)

/** Expose C++ Object With the Given functions **/
RCPP_MODULE(MMPRTile) {


    void (MPRTile::*pChangePrecision)(const size_t &, const size_t &,
                                      const std::string &) =&MPRTile::ChangePrecision;

    void (MPRTile::*pFillTriangle)(const double &, const bool &,
                                   const std::string &) =&MPRTile::FillSquareTriangle;

    using namespace Rcpp;
    class_ <MPRTile>("MMPRTile")
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
        .method("MMPRTile.SetVal", &MPRTile::SetVal)
        .method("MMPRTile.GetVal", &MPRTile::GetVal)
        .method("show", &MPRTile::GetType)
        .method("MMPRTile.print", &MPRTile::Print)
        .method("FillSquareTriangle", pFillTriangle)
        .method("Sum", &MPRTile::Sum)
        .method("Prod", &MPRTile::Product);

    /** MPRTile Functions **/
    function("MMPRTile.copy",&RCopyMPRTile,List::create(_["x"]));
    function("MMPRTile.gemm", &mpr::operations::linear::TileGemm,
             List::create(_[ "a" ], _[ "b" ], _[ "c" ],
                          _[ "transpose_a" ] = false,
                          _[ "transpose_b" ] = false, _[ "alpha" ] = 1,
                          _[ "beta" ] = 0));

    function("MMPRTile.chol", &mpr::operations::linear::TileCholesky,
             List::create(_[ "x" ], _[ "overwrite_input" ] = true,
                          _[ "num_threads" ] = 1));

    function("MMPRTile.trsm", &mpr::operations::linear::TileTrsm,
             List::create(_[ "a" ], _[ "b" ], _[ "side" ],
                          _[ "upper_triangle" ], _[ "transpose" ],
                          _[ "alpha" ]));

    function("MMPRTile.GetTile", &RGetTile,
             List::create(_[ "matrix" ], _[ "row" ], _[ "col" ]));
    function("MMPRTile.UpdateTile", &RInsertTile,
             List::create(_[ "matrix" ], _[ "tile" ], _[ "row" ], _[ "col" ]));


}

