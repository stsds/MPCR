/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <libraries/catch/catch.hpp>
#include <utilities/MPCRDispatcher.hpp>
#include <operations/helpers/LinearAlgebraHelper.hpp>
#include <operations/LinearAlgebra.hpp>
#include <data-units/DataType.hpp>
#include <operations/cuda/CudaHelpers.hpp>


using namespace std;
using namespace mpcr::precision;
using namespace mpcr::operations;


void
TEST_LINEAR_ALGEBRA_HELPERS() {

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


        Symmetrize <float>(a, true);
        auto is_symmetric = false;
        linear::IsSymmetric <float>(a, is_symmetric);

        REQUIRE(is_symmetric == true);


        b.GetData(GPU);
        helpers::CudaHelpers::Symmetrize <float>(b, true,
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

        Symmetrize <float>(c, false);
        is_symmetric = false;
        linear::IsSymmetric <float>(c, is_symmetric);


        d.GetData(GPU);
        helpers::CudaHelpers::Symmetrize <float>(d, false,
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

        a.FillTriangle(0, true);

        b.GetData(GPU);
        helpers::CudaHelpers::FillTriangle <float>(b, 0, true,
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


        c.FillTriangle(0, false);
        helpers::CudaHelpers::FillTriangle <float>(d, 0, false,
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

        helpers::CudaHelpers::Reverse <float>(b,
                                              mpcr::kernels::ContextManager::GetGPUContext());

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

        ReverseMatrix <float>(a);

        helpers::CudaHelpers::Reverse <float>(b,
                                              mpcr::kernels::ContextManager::GetGPUContext());

        for (auto i = 0; i < values.size(); i++) {
            REQUIRE(a.GetVal(i) == b.GetVal(i));
        }
    }SECTION("Transpose"){

        vector <double> values;
        auto row = 10;
        auto col=20;
        auto size=row*col;
        values.resize(size);
        for (auto i = 0; i < size; i++) {
            values[ i ] = i;
        }

        DataType a(values, FLOAT);
        DataType b(values, FLOAT);
        a.ToMatrix(row, col);
        b.ToMatrix(row, col);

        a.Transpose();
        REQUIRE(a.GetNRow()==col);
        REQUIRE(a.GetNCol()==row);
        REQUIRE(a.GetSize()==size);

        helpers::CudaHelpers::Transpose<float>(b,
                                              mpcr::kernels::ContextManager::GetGPUContext());

        REQUIRE(b.GetNRow()==col);
        REQUIRE(b.GetNCol()==row);
        REQUIRE(b.GetSize()==size);

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
