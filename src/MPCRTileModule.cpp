/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <data-units/MPCRTile.hpp>
#include <operations/TileLinearAlgebra.hpp>
#include <adapters/RHelpers.hpp>

/** Expose C++ class to R to be able to use Wrap and As
 *  Allows C++ to Send and Receive Class object from R
 **/
RCPP_EXPOSED_CLASS(MPCRTile)
RCPP_EXPOSED_CLASS(DataType)

/** Expose C++ Object With the Given functions **/
RCPP_MODULE(MPCRTile) {


    void (MPCRTile::*pChangePrecision)(const size_t &, const size_t &,
                                      const std::string &) =&MPCRTile::ChangePrecision;

    void (MPCRTile::*pFillTriangle)(const double &, const bool &,
                                   const std::string &) =&MPCRTile::FillSquareTriangle;

    using namespace Rcpp;
    class_ <MPCRTile>("MPCRTile")
        .constructor <size_t, size_t, size_t, size_t,
            std::vector <double>, std::vector <std::string> >()
        .property("Row", &MPCRTile::GetNRow)
        .property("Col", &MPCRTile::GetNCol)
        .property("Size", &MPCRTile::GetMatrixSize)
        .property("TileRow", &MPCRTile::GetTileNRow)
        .property("TileCol", &MPCRTile::GetTileNCol)
        .property("TileSize", &MPCRTile::GetTileSize)
        .method("PrintTile", &MPCRTile::PrintTile)
        .method("ChangeTilePrecision", pChangePrecision)
        .method("MPCRTile.SetVal", &MPCRTile::SetVal)
        .method("MPCRTile.GetVal", &MPCRTile::GetVal)
        .method("show", &MPCRTile::GetType)
        .method("MPCRTile.print", &MPCRTile::Print)
        .method("FillSquareTriangle", pFillTriangle)
        .method("Diag", &MPCRTile::GetDiagonal)
        .method("Sum", &MPCRTile::Sum)
        .method("SquareSum", &MPCRTile::SquareSum)
        .method("Norm", &MPCRTile::Norm)
        .method("Prod", &MPCRTile::Product);

    /** MPCRTile Functions **/
    function("MPCRTile.copy",&RCopyMPCRTile,List::create(_["x"]));
#ifdef MPCR_INSTALL
    function("MPCRTile.gemm", &mpcr::operations::linear::TileGemm,
             List::create(_[ "a" ], _[ "b" ], _[ "c" ],
                          _[ "transpose_a" ] = false,
                          _[ "transpose_b" ] = false, _[ "alpha" ] = 1,
                          _[ "beta" ] = 0,_[ "num_threads" ] = 1));

    function("MPCRTile.chol", &mpcr::operations::linear::TileCholesky,
             List::create(_[ "x" ], _[ "overwrite_input" ] = true,
                          _[ "num_threads" ] = 1));

    function("MPCRTile.trsm", &mpcr::operations::linear::TileTrsm,
             List::create(_[ "a" ], _[ "b" ], _[ "side" ],
                          _[ "upper_triangle" ], _[ "transpose" ],
                          _[ "alpha" ]));
#endif
    function("MPCRTile.GetTile", &RGetTile,
             List::create(_[ "matrix" ], _[ "row" ], _[ "col" ]));
    function("MPCRTile.UpdateTile", &RInsertTile,
             List::create(_[ "matrix" ], _[ "tile" ], _[ "row" ], _[ "col" ]));

    function("MPCRTile.GetSerializedTile", &RGetSerializeTile,
             List::create(_[ "matrix" ], _[ "row" ], _[ "col" ]));


}
