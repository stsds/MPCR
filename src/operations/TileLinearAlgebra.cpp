
#include <operations/TileLinearAlgebra.hpp>
#include <operations/LinearAlgebra.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <data-units/Promoter.hpp>


using namespace mpr::operations;


MPRTile *
linear::TileCholesky(MPRTile &aMatrix, const bool &aOverWriteInput) {

    auto tiles_per_row = aMatrix.GetTilePerRow();
    auto tiles_per_col = aMatrix.GetTilePerCol();

    if (tiles_per_row != tiles_per_col) {
        MPR_API_EXCEPTION(
            "Cannot perform Cholesky decomposition on non square tiled MPRTile object",
            -1);
    }

    MPRTile *pOutput = nullptr;
    if (aOverWriteInput) {
        pOutput = &aMatrix;
    } else {
        pOutput = new MPRTile(aMatrix);
    }

    Promoter prom(2);

    for (auto k = 0; k < tiles_per_row; k++) {

        auto pTemp_tile_out = new DataType(FLOAT);
        auto pTile_matrix_a = pOutput->GetTile(k, k);

        prom.ResetPromoter(2);
        prom.Insert(*pTemp_tile_out);
        prom.Insert(*pTile_matrix_a);
        prom.Promote();

        SIMPLE_DISPATCH(pTemp_tile_out->GetPrecision(), linear::Cholesky,
                        *pTile_matrix_a, *pTemp_tile_out, false)

        prom.DePromote();

        pOutput->InsertTile(pTemp_tile_out, k, k);

        for (auto i = k + 1; i < tiles_per_row; i++) {

            auto pTemp_tile_out_two = new DataType(FLOAT);
            auto pTemp_tile_a = pOutput->GetTile(i, k);
            auto pTemp_tile_b = pOutput->GetTile(k, k);

            prom.ResetPromoter(3);
            prom.Insert(*pTemp_tile_b);
            prom.Insert(*pTemp_tile_out_two);
            prom.Insert(*pTemp_tile_a);

            prom.Promote();

            SIMPLE_DISPATCH(pTemp_tile_out->GetPrecision(),
                            linear::BackSolve,
                            *pTemp_tile_b, *pTemp_tile_a,
                            *pTemp_tile_out_two,
                            pTemp_tile_b->GetNCol(), false, true, 'R')

            prom.DePromote();

            pOutput->InsertTile(pTemp_tile_out_two, i, k);

        }

        for (auto j = k + 1; j < tiles_per_row; j++) {

            pTile_matrix_a = pOutput->GetTile(j, k);
            auto pTemp_tile_out_two = pOutput->GetTile(j, j);
            DataType dump(pTile_matrix_a->GetPrecision());

            prom.ResetPromoter(2);
            prom.Insert(*pTile_matrix_a);
            prom.Insert(*pTemp_tile_out_two);
            prom.Promote();

            SIMPLE_DISPATCH(pTemp_tile_out->GetPrecision(),
                            linear::CrossProduct, *pTile_matrix_a, dump,
                            *pTemp_tile_out_two, false, false, false, -1, 1)

            prom.DePromote();
            pOutput->InsertTile(pTemp_tile_out_two, j, j);

            for (auto i = j + 1; i < tiles_per_row; i++) {

                auto pTile_matrix_a_two = pOutput->GetTile(i, k);
                auto pTemp_tile_out_temp = pOutput->GetTile(i, j);

                prom.ResetPromoter(3);
                prom.Insert(*pTemp_tile_out_temp);
                prom.Insert(*pTile_matrix_a_two);
                prom.Insert(*pTile_matrix_a);

                prom.Promote();


                SIMPLE_DISPATCH(pTemp_tile_out->GetPrecision(),
                                linear::CrossProduct, *pTile_matrix_a_two,
                                *pTile_matrix_a, *pTemp_tile_out_temp,
                                false, true, true, -1, 1)

                prom.DePromote();
                pOutput->InsertTile(pTemp_tile_out_temp, i, j);
            }
        }
    }
    pOutput->FillSquareTriangle(0, true);

    return pOutput;
}


MPRTile *
linear::TileGemm(MPRTile &aInputA, MPRTile &aInputB) {
    auto tile_per_row_a = aInputA.GetTilePerRow();
    auto tile_per_col_a = aInputA.GetTilePerCol();

    auto tile_per_row_b = aInputB.GetTilePerRow();
    auto tile_per_col_b = aInputB.GetTilePerCol();


    if (tile_per_col_a != tile_per_row_b) {
        MPR_API_EXCEPTION(
            "Cannot perform Matrix multiplication, Tiles Per Col A != Tiles Per Row B",
            -1);
    }

    Dimensions internal(aInputA.GetTileNRow(), aInputB.GetTileNCol());

    auto pOutput = new MPRTile(aInputA.GetNRow(), aInputB.GetNCol(),
                               internal.GetNRow(), internal.GetNCol());


    for (auto i = 0; i < tile_per_row_a; i++) {
        for (auto j = 0; j < tile_per_col_b; j++) {

            auto pTile_c = new DataType(FLOAT);
            Promoter prom(3);

            for (auto k = 0; k < tile_per_col_a; k++) {
                auto *pTile_a = aInputA.GetTile(i, k);
                auto *pTile_b = aInputB.GetTile(k, j);

                prom.Insert(*pTile_a);
                prom.Insert(*pTile_b);
                prom.Insert(*pTile_c);
                prom.Promote();

                SIMPLE_DISPATCH(pTile_c->GetPrecision(), linear::CrossProduct,
                                *pTile_a, *pTile_b, *pTile_c, false, false,
                                true, 1, 1)

                prom.DePromote();
                prom.ResetPromoter(3);

            }

            pOutput->InsertTile(pTile_c, i, j);
        }
    }

    return pOutput;
}