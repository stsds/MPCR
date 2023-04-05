
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
    Promoter dep_promoter(1);

    for (auto k = 0; k < tiles_per_row; k++) {

        auto pTile_matrix_a_potrf = pOutput->GetTile(k, k);
        auto pTile_potrf_out = new DataType(
            pTile_matrix_a_potrf->GetPrecision());

        prom.ResetPromoter(2);
        prom.Insert(*pTile_matrix_a_potrf);
        prom.Insert(*pTile_potrf_out);
        prom.Promote();

        /** potrf **/
        SIMPLE_DISPATCH(pTile_potrf_out->GetPrecision(), linear::Cholesky,
                        *pTile_matrix_a_potrf, *pTile_potrf_out, false)

        prom.ResetPromoter(2);

        pOutput->InsertTile(pTile_potrf_out, k, k);

#pragma omp parallel for
        for (auto i = k + 1; i < tiles_per_row; i++) {

            Promoter temp_promoter(2);

            auto pTemp_tile_a = pOutput->GetTile(k, k);
            auto pTemp_tile_b = pOutput->GetTile(i, k);
            auto pTile_trsm_out = new DataType(
                pTemp_tile_b->GetPrecision());

            temp_promoter.Insert(*pTile_trsm_out);
            temp_promoter.Insert(*pTemp_tile_b);

            temp_promoter.Promote();
            auto pTile_a_promoted = dep_promoter.GetPromotedTile(pTemp_tile_a,
                                                                 pTemp_tile_b->GetPrecision());


            /** trsm **/
            SIMPLE_DISPATCH(pTemp_tile_b->GetPrecision(), linear::BackSolve,
                            *pTile_a_promoted, *pTemp_tile_b,
                            *pTile_trsm_out, pTile_a_promoted->GetNCol(),
                            false,
                            true, 'R')

            temp_promoter.DePromote();
            pOutput->InsertTile(pTile_trsm_out, i, k);

        }
        dep_promoter.ResetPromoter(1);
        for (auto j = k + 1; j < tiles_per_row; j++) {

            auto pTile_matrix_a = pOutput->GetTile(j, k);
            auto pTemp_tile_out_two = pOutput->GetTile(j, j);

            prom.ResetPromoter(1);
            prom.Insert(*pTemp_tile_out_two);
            prom.Promote();

            DataType dump(pTemp_tile_out_two->GetPrecision());


            auto pTile_a_promoted = dep_promoter.GetPromotedTile(pTile_matrix_a,
                                                                 pTemp_tile_out_two->GetPrecision());




            /** syrk **/
            SIMPLE_DISPATCH(pTile_a_promoted->GetPrecision(),
                            linear::CrossProduct, *pTile_a_promoted, dump,
                            *pTemp_tile_out_two, false, false, false, -1, 1)

            prom.DePromote();
            pOutput->InsertTile(pTemp_tile_out_two, j, j);

#pragma omp parallel for
            for (auto i = j + 1; i < tiles_per_row; i++) {

                auto pTemp_tile_a = pOutput->GetTile(i, k);
                auto pTemp_tile_b = pOutput->GetTile(j, k);
                auto pTile_gemm_out = pOutput->GetTile(i, j);

                Promoter temp_promoter(1);
                temp_promoter.ResetPromoter(1);
                temp_promoter.Insert(*pTile_gemm_out);
                temp_promoter.Promote();

                auto pTile_a_promoted_gemm = dep_promoter.GetPromotedTile(
                    pTemp_tile_a, pTile_gemm_out->GetPrecision());
                auto pTile_b_promoted = dep_promoter.GetPromotedTile(
                    pTemp_tile_b, pTile_gemm_out->GetPrecision());

                /** gemm **/
                SIMPLE_DISPATCH(pTile_gemm_out->GetPrecision(),
                                linear::CrossProduct, *pTile_a_promoted_gemm,
                                *pTile_b_promoted, *pTile_gemm_out,
                                false, true, true, -1, 1)

                temp_promoter.DePromote();
                pOutput->InsertTile(pTile_gemm_out, i, j);
            }
            dep_promoter.ResetPromoter(1);
        }
    }
    pOutput->FillSquareTriangle(0, true);

    return pOutput;
}


MPRTile *
linear::TileGemm(MPRTile &aInputA, MPRTile &aInputB, MPRTile &aInputC,
                 const double &aAlpha, const double &aBeta) {
    auto tile_per_row_a = aInputA.GetTilePerRow();
    auto tile_per_col_a = aInputA.GetTilePerCol();

    auto tile_per_row_b = aInputB.GetTilePerRow();
    auto tile_per_col_b = aInputB.GetTilePerCol();


    auto tile_per_row_c = aInputC.GetTilePerRow();
    auto tile_per_col_c = aInputC.GetTilePerCol();


    if (tile_per_col_a != tile_per_row_b || tile_per_row_a != tile_per_row_c ||
        tile_per_col_b != tile_per_col_c) {
        MPR_API_EXCEPTION(
            "Cannot perform Matrix multiplication, Tiles Per Col A != Tiles Per Row B",
            -1);
    }


    auto pOutput = &aInputC;
    Promoter dep_promoter(1);
    Promoter prom(1);


    for (auto i = 0; i < tile_per_row_a; i++) {
        for (auto j = 0; j < tile_per_col_b; j++) {

            auto pTile_c = pOutput->GetTile(i, j);
            prom.ResetPromoter(1);
            prom.Insert(*pTile_c);
            prom.Promote();

            for (auto k = 0; k < tile_per_col_a; k++) {
                Promoter temp_dep_prom(1);
                auto *pTile_a = aInputA.GetTile(i, k);
                auto *pTile_b = aInputB.GetTile(k, j);

                auto pTile_a_promoted = dep_promoter.GetPromotedTile(pTile_a,
                                                                     pTile_c->GetPrecision());
                auto pTile_b_promoted = temp_dep_prom.GetPromotedTile(pTile_b,
                                                                     pTile_c->GetPrecision());


                SIMPLE_DISPATCH(pTile_c->GetPrecision(), linear::CrossProduct,
                                *pTile_a_promoted, *pTile_b_promoted, *pTile_c, false, false,
                                true, aAlpha, aBeta)

            }
            prom.DePromote();
            pOutput->InsertTile(pTile_c, i, j);
        }
        dep_promoter.ResetPromoter(1);
    }

    return pOutput;
}