
#include <operations/TileLinearAlgebra.hpp>
#include <operations/LinearAlgebra.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <data-units/Promoter.hpp>


using namespace mpr::operations;


MPRTile *
linear::TileCholesky(MPRTile &aMatrix) {

    auto tiles_per_row = aMatrix.GetTilePerRow();
    auto tiles_per_col = aMatrix.GetTilePerCol();

    if (tiles_per_row != tiles_per_col) {
        MPR_API_EXCEPTION(
            "Cannot perform Cholesky decomposition on non square tiled MPRTile object",
            -1);
    }

    auto pOutput = new MPRTile(aMatrix.GetNRow(), aMatrix.GetNCol(),
                               aMatrix.GetTileNRow(), aMatrix.GetTileNCol());

    MPRTile temp_input(aMatrix);

    Promoter prom(2);


    for (auto k = 0; k < tiles_per_row; k++) {

        auto pTemp_tile = new DataType(FLOAT);
        auto pTile_matrix_a = temp_input.GetTile(k, k);

        prom.ResetPromoter(2);
        prom.Insert(*pTemp_tile);
        prom.Insert(*pTile_matrix_a);
        prom.Promote();

        SIMPLE_DISPATCH(pTemp_tile->GetPrecision(), linear::Cholesky,
                        *pTile_matrix_a, *pTemp_tile, false)


        prom.DePromote();
        std::cout << "tile k,k " << k << "  " << k << std::endl;
        pOutput->InsertTile(pTemp_tile, k, k);

        for (auto i = k + 1; i < tiles_per_row; i++) {

            auto pTemp_tile_two = new DataType(FLOAT);
            pTile_matrix_a = temp_input.GetTile(i, k);

            prom.ResetPromoter(3);
            prom.Insert(*pTemp_tile);
            prom.Insert(*pTemp_tile_two);
            prom.Insert(*pTile_matrix_a);
            prom.Promote();

            SIMPLE_DISPATCH(pTemp_tile_two->GetPrecision(), linear::BackSolve,
                            *pTemp_tile, *pTile_matrix_a, *pTemp_tile_two,
                            pTemp_tile->GetNCol(), false, false)


            prom.DePromote();
            std::cout << "tile i,k " << i << "  " << k << std::endl;
            pOutput->InsertTile(pTemp_tile_two, i, k);

        }

        for (auto j = k + 1; j < tiles_per_row; j++) {

            pTile_matrix_a = pOutput->GetTile(j, k);
            auto pTemp_tile_two = new DataType(pTile_matrix_a->GetPrecision());
            DataType dump(pTile_matrix_a->GetPrecision());

            prom.ResetPromoter(2);
            prom.Insert(*pTile_matrix_a);
            prom.Insert(*pTemp_tile_two);
            prom.Promote();

            SIMPLE_DISPATCH(pTemp_tile_two->GetPrecision(),
                            linear::CrossProduct, *pTile_matrix_a, dump,
                            *pTemp_tile_two, false, false, false)

            prom.DePromote();
            std::cout << "tile j,j " << j << "  " << j << std::endl;
            temp_input.InsertTile(pTemp_tile_two, j, j);

            for (auto i = j + 1; j < tiles_per_row; j++) {

                if (i >= tiles_per_row) {
                    break;
                }

                auto pTemp_tile_three = new DataType(
                    pTile_matrix_a->GetPrecision());
                auto pTile_matrix_a_two = pOutput->GetTile(i, k);

                prom.ResetPromoter(3);
                prom.Insert(*pTemp_tile_three);
                prom.Insert(*pTile_matrix_a_two);
                prom.Insert(*pTile_matrix_a);
                prom.Promote();

                SIMPLE_DISPATCH(pTemp_tile_three->GetPrecision(),
                                linear::CrossProduct, *pTile_matrix_a_two,
                                *pTile_matrix_a, *pTemp_tile_three, false,
                                false)

                prom.DePromote();
                std::cout << "tile i,j " << i << "  " << j << std::endl;
                temp_input.InsertTile(pTemp_tile_three, i, j);

            }

        }

    }
    pOutput->FillWithZeros();

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

    Promoter prom(4);


    for (auto i = 0; i < tile_per_row_a; i++) {
        for (auto j = 0; j < tile_per_col_b; j++) {
            auto pTile_c = new DataType(FLOAT);
            for (auto k = 0; k < tile_per_col_a; k++) {
                auto pTile_temp = new DataType(FLOAT);
                auto pTile_a = aInputA.GetTile(i, k);
                auto pTile_b = aInputB.GetTile(k, j);
                prom.Insert(*pTile_temp);
                prom.Insert(*pTile_a);
                prom.Insert(*pTile_b);
                prom.Insert(*pTile_c);
                prom.Promote();

                SIMPLE_DISPATCH(pTile_c->GetPrecision(), linear::CrossProduct,
                                *pTile_a, *pTile_b, *pTile_temp, false, false)


                SIMPLE_DISPATCH(pTile_c->GetPrecision(), linear::AddTiles,
                                *pTile_temp, *pTile_c)

                prom.DePromote();
                delete pTile_temp;
                prom.ResetPromoter(4);

            }
            pOutput->InsertTile(pTile_c, i, j);
        }
    }

    return pOutput;
}


template <typename T>
void
linear::AddTiles(DataType &aInputA, DataType &aInputB) {

    auto size_a = aInputA.GetSize();

    if (aInputB.GetSize() == 0) {
        aInputB.SetSize(size_a);
        auto data = new T[size_a];
        memset((void *) data, 0, size_a * sizeof(T));
        aInputB.SetData((char *) data);
        aInputB.SetDimensions(aInputA);
    }

    auto size_b = aInputB.GetSize();
    if (size_a != size_b) {
        MPR_API_EXCEPTION("Cannot add two tiles, not the same size", -1);
    }

    auto data_a = (T *) aInputA.GetData();
    auto data_b = (T *) aInputB.GetData();

    for (auto i = 0; i < size_a; i++) {
        data_b[ i ] += data_a[ i ];
    }
}


FLOATING_POINT_INST(void, linear::AddTiles, DataType &aInputA,
                    DataType &aInputB)

