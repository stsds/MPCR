/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <libraries/catch/catch.hpp>
#include <utilities/MPCRDispatcher.hpp>
#include <operations/concrete/BackendFactory.hpp>
#include <operations/LinearAlgebra.hpp>


using namespace std;
using namespace mpcr::precision;
using namespace mpcr::operations;


void
TEST_LINEAR_ALGEBRA_HELPERS() {
    auto helper_host = BackendFactory <float>::CreateHelpersBackend(CPU);
    auto helper_dev = BackendFactory <float>::CreateHelpersBackend(GPU);

    SECTION("Symmetrize") {

        cout << "Testing Symmetrize ..." << endl;
        vector <double> values;
        auto side_len = 100;
        values.resize(side_len * side_len);
        for (auto i = 0; i < side_len * side_len; i++) {
            values[ i ] = i;
        }
        DataType a(values, FLOAT);
        DataType b(values, FLOAT);
        a.ToMatrix(side_len, side_len);
        b.ToMatrix(side_len, side_len);


        helper_host->Symmetrize(a, true, nullptr);
        auto is_symmetric = false;
        linear::IsSymmetric <float>(a, is_symmetric);

        REQUIRE(is_symmetric == true);


        b.GetData(GPU);
        helper_dev->Symmetrize(b, true,
                               mpcr::kernels::ContextManager::GetGPUContext());

        b.GetData(CPU);

        is_symmetric = false;
        linear::IsSymmetric <float>(b, is_symmetric);

        REQUIRE(is_symmetric == true);

        for (auto i = 0; i < side_len * side_len; i++) {
            REQUIRE(a.GetVal(i) == b.GetVal(i));
        }


        DataType c(values, FLOAT);
        DataType d(values, FLOAT);

        c.ToMatrix(side_len, side_len);
        d.ToMatrix(side_len, side_len);

        helper_host->Symmetrize(c, false, nullptr);
        is_symmetric = false;
        linear::IsSymmetric <float>(c, is_symmetric);


        d.GetData(GPU);
        helper_dev->Symmetrize(d, false,
                               mpcr::kernels::ContextManager::GetGPUContext());

        d.GetData(CPU);


        is_symmetric = false;
        linear::IsSymmetric <float>(d, is_symmetric);

        REQUIRE(is_symmetric == true);

        for (auto i = 0; i < side_len * side_len; i++) {
            REQUIRE(c.GetVal(i) == d.GetVal(i));
        }

    }SECTION("Fill Triangle") {

        vector <double> values;
        auto side_len = 100;
        values.resize(side_len * side_len);
        for (auto i = 0; i < side_len * side_len; i++) {
            values[ i ] = i;
        }

        DataType a(values, FLOAT);
        DataType b(values, FLOAT);
        a.ToMatrix(side_len, side_len);
        b.ToMatrix(side_len, side_len);

        helper_host->FillTriangle(a, 0, true, nullptr);


        b.GetData(GPU);
        helper_dev->FillTriangle(b, 0, true,
                                 mpcr::kernels::ContextManager::GetGPUContext());

        b.GetData(CPU);
        REQUIRE(b.GetSize() == a.GetSize());
        REQUIRE(b.GetNCol() == a.GetNCol());
        REQUIRE(b.GetNRow() == a.GetNRow());

        for (auto i = 0; i < side_len * side_len; i++) {
            REQUIRE(a.GetVal(i) == b.GetVal(i));
        }

        DataType c(values, FLOAT);
        DataType d(values, FLOAT, GPU);

        c.ToMatrix(side_len, side_len);
        d.ToMatrix(side_len, side_len);


        helper_host->FillTriangle(c, 0, false, nullptr);

        helper_dev->FillTriangle(d, 0, false,
                                 mpcr::kernels::ContextManager::GetGPUContext());

        REQUIRE(c.GetSize() == d.GetSize());
        REQUIRE(c.GetNCol() == d.GetNCol());
        REQUIRE(c.GetNRow() == d.GetNRow());


        for (auto i = 0; i < side_len * side_len; i++) {
            REQUIRE(c.GetVal(i) == d.GetVal(i));
        }


    }SECTION("Reverse Array") {
        vector <double> values;
        auto side_len = 100;
        values.resize(side_len * side_len);
        for (auto i = 0; i < side_len * side_len; i++) {
            values[ i ] = i;
        }

        DataType a(values, FLOAT);
        DataType b(values, FLOAT);

        helper_dev->Reverse(b, mpcr::kernels::ContextManager::GetGPUContext());

        for (auto i = 0; i < values.size(); i++) {
            REQUIRE(a.GetVal(values.size() - i - 1) == b.GetVal(i));
        }

    }SECTION("Reverse Matrix") {

        vector <double> values;
        auto side_len = 100;
        values.resize(side_len * side_len);
        for (auto i = 0; i < side_len * side_len; i++) {
            values[ i ] = i;
        }

        DataType a(values, FLOAT);
        DataType b(values, FLOAT);
        a.ToMatrix(side_len, side_len);
        b.ToMatrix(side_len, side_len);

        helper_host->Reverse(a, nullptr);
        helper_dev->Reverse(b, mpcr::kernels::ContextManager::GetGPUContext());

        for (auto i = 0; i < values.size(); i++) {
            REQUIRE(a.GetVal(i) == b.GetVal(i));
        }
    }SECTION("Transpose") {

        vector <double> values;
        auto row = 10;
        auto col = 20;
        auto size = row * col;
        values.resize(size);
        for (auto i = 0; i < size; i++) {
            values[ i ] = i;
        }

        DataType a(values, FLOAT);
        DataType b(values, FLOAT);
        a.ToMatrix(row, col);
        b.ToMatrix(row, col);

        helper_host->Transpose(a, nullptr);
        REQUIRE(a.GetNRow() == col);
        REQUIRE(a.GetNCol() == row);
        REQUIRE(a.GetSize() == size);

        helper_dev->Transpose(b,
                              mpcr::kernels::ContextManager::GetGPUContext());

        REQUIRE(b.GetNRow() == col);
        REQUIRE(b.GetNCol() == row);
        REQUIRE(b.GetSize() == size);

        for (auto i = 0; i < values.size(); i++) {
            REQUIRE(a.GetVal(i) == b.GetVal(i));
        }
    }


}


TEST_CASE("LinearAlgebraHelpers", "[Linear Algebra Helpers]") {
    mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
        CPU);
    TEST_LINEAR_ALGEBRA_HELPERS();

}