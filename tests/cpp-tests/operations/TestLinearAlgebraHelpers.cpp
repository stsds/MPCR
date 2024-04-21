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
    }SECTION("NORM") {

        DataType mat_host(6, 6, FLOAT, CPU);
        DataType mat_dev(6, 6, FLOAT);

        for (auto i = 0; i < mat_host.GetSize(); i++) {
            mat_host.SetVal(i, i * 2);
            mat_dev.SetVal(i, i * 2);
        }
        mat_dev.GetData(GPU);
        mat_dev.FreeMemory(CPU);

        REQUIRE(mat_dev.GetSize() != 0);

        float out_host = 0;
        float out_dev = 0;

        helper_host->NormMACS(mat_host, out_host, nullptr);
        helper_dev->NormMACS(mat_dev, out_dev,
                             mpcr::kernels::ContextManager::GetGPUContext());

        REQUIRE(out_host != 0);
        REQUIRE(out_host == out_dev);


        out_host = 0;
        out_dev = 0;

        helper_host->NormMARS(mat_host, out_host, nullptr);
        helper_dev->NormMARS(mat_dev, out_dev,
                             mpcr::kernels::ContextManager::GetGPUContext());

        REQUIRE(out_host != 0);
        REQUIRE(out_host == out_dev);

        out_host = 0;
        out_dev = 0;

        helper_host->NormMaxMod(mat_host, out_host, nullptr);
        helper_dev->NormMaxMod(mat_dev, out_dev,
                               mpcr::kernels::ContextManager::GetGPUContext());


        REQUIRE(out_host != 0);
        REQUIRE(out_host == out_dev);

        out_host = 0;
        out_dev = 0;

        helper_host->NormEuclidean(mat_host, out_host, nullptr);
        helper_dev->NormEuclidean(mat_dev, out_dev,
                                  mpcr::kernels::ContextManager::GetGPUContext());

        REQUIRE(out_host != 0);
        REQUIRE(out_host == out_dev);

    }SECTION("Get Rank") {
        DataType mat_host(6, 6, FLOAT, CPU);
        DataType mat_dev(6, 6, FLOAT);

        for (auto i = 0; i < mat_host.GetSize(); i++) {
            mat_host.SetVal(i, i * 2);
            mat_dev.SetVal(i, i * 2);
        }

        for (auto i = 0; i < mat_host.GetNRow(); i++) {
            mat_host.SetValMatrix(i, i, i);
            mat_dev.SetValMatrix(i, i, i);
        }


        mat_dev.GetData(GPU);
        mat_dev.FreeMemory(CPU);

        float out_host = 0;
        float out_dev = 0;
        double tolerance = 1e-10;
        helper_host->GetRank(mat_host, tolerance, out_host, nullptr);
        helper_dev->GetRank(mat_dev, tolerance, out_dev,
                            mpcr::kernels::ContextManager::GetGPUContext());

        REQUIRE(out_host!=0);
        REQUIRE(out_dev==out_host);
    }SECTION("IsSymmetric"){
        vector <double> values_symmetric = {2, 3, 6, 3, 4, 5, 6, 5, 9};
        vector <double> values_non_symmetric = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        DataType a_symmetric(values_symmetric, FLOAT);
        a_symmetric.ToMatrix(3, 3);


        DataType b_non_symmetric(values_non_symmetric, FLOAT);
        b_non_symmetric.ToMatrix(3, 3);

        auto output=false;
        helper_host->IsSymmetric(a_symmetric,output, nullptr);
        REQUIRE(output==true);


        helper_host->IsSymmetric(b_non_symmetric,output, nullptr);
        REQUIRE(output==false);

        auto output_dev=false;
        helper_dev->IsSymmetric(a_symmetric,output_dev, mpcr::kernels::ContextManager::GetGPUContext());
        REQUIRE(output_dev==true);


        helper_dev->IsSymmetric(b_non_symmetric,output_dev, mpcr::kernels::ContextManager::GetGPUContext());
        REQUIRE(output_dev==false);

    }


}


TEST_CASE("LinearAlgebraHelpers", "[Linear Algebra Helpers]") {
    mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
        CPU);
    TEST_LINEAR_ALGEBRA_HELPERS();

}
