/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include "libraries/catch/catch.hpp"
#include "utilities/MPCRDispatcher.hpp"
#include "operations/LinearAlgebra.hpp"


using namespace mpcr::operations;
using namespace mpcr::precision;
using namespace mpcr::kernels;
using namespace std;


void
TEST_CUDA_STREAMS() {

    SECTION("Test Asynchronous GPU context without synchronization") {
        cout << "Testing CUDA streams ..." << endl;

        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        auto &context_manager = mpcr::kernels::ContextManager::GetInstance();
        auto default_context = mpcr::kernels::ContextManager::GetOperationContext();

        auto gpu_context_async = context_manager.CreateRunContext("GPU");
        gpu_context_async->SetOperationPlacement(GPU);
        gpu_context_async->SetRunMode( RunMode::ASYNC);

        const int size = 1024;
        std::vector<double> values(size * size, 1.0);

        DataType a(values, DOUBLE, GPU);
        a.ToMatrix(size, size);

        DataType b(values, DOUBLE, GPU);
        b.ToMatrix(size, size);

        DataType output_validate(DOUBLE, GPU);
        DataType output(DOUBLE, GPU);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_validate, false,false)

        context_manager.SetOperationContext(gpu_context_async);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output, false,false)

        context_manager.SetOperationContext(default_context);

        auto data = (double *) output.GetData();
        auto data_validate = (double *) output_validate.GetData();
        REQUIRE(output.GetNRow() == size);
        REQUIRE(output.GetNCol() == size);

        double error_threshold = 0.001;
        for (int i = 0; i < output.GetSize(); ++i) {
            double val = fabs(data[i] - data_validate[i]) / data_validate[i];
            REQUIRE(val > error_threshold);
        }
        mpcr::kernels::ContextManager::DestroyInstance();

    }

    SECTION("Test Asynchronous GPU context with synchronization") {

        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        auto &context_manager = mpcr::kernels::ContextManager::GetInstance();
        auto default_context = mpcr::kernels::ContextManager::GetOperationContext();

        auto gpu_context_async = context_manager.CreateRunContext("GPU");
        gpu_context_async->SetOperationPlacement(GPU);
        gpu_context_async->SetRunMode( RunMode::ASYNC);

        const int size = 1024;
        std::vector<double> values(size * size, 1.0);

        DataType a(values, DOUBLE, GPU);
        a.ToMatrix(size, size);

        DataType b(values, DOUBLE, GPU);
        b.ToMatrix(size, size);

        DataType output_validate(DOUBLE, GPU);
        DataType output(DOUBLE, GPU);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_validate, false,false)

        context_manager.SetOperationContext(gpu_context_async);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output, false,false)

        context_manager.SetOperationContext(default_context);
        context_manager.SyncContext("GPU");

        auto data = (double *) output.GetData();
        auto data_validate = (double *) output_validate.GetData();
        REQUIRE(output.GetNRow() == size);
        REQUIRE(output.GetNCol() == size);

        double error_threshold = 0.001;
        for (int i = 0; i < output.GetSize(); ++i) {
            double val = fabs(data[i] - data_validate[i]) / data_validate[i];
            REQUIRE(val <= error_threshold);
        }
        mpcr::kernels::ContextManager::DestroyInstance();
    }
    SECTION("Test Synchronous GPU context") {

        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        auto &context_manager = mpcr::kernels::ContextManager::GetInstance();
        auto default_context = mpcr::kernels::ContextManager::GetOperationContext();

        auto gpu_context_sync = context_manager.CreateRunContext("GPU");
        gpu_context_sync->SetOperationPlacement(GPU);
        gpu_context_sync->SetRunMode( RunMode::SYNC);

        const int size = 1024;
        std::vector<double> values(size * size, 1.0);

        DataType a(values, DOUBLE, GPU);
        a.ToMatrix(size, size);

        DataType b(values, DOUBLE, GPU);
        b.ToMatrix(size, size);

        DataType output_validate(DOUBLE, GPU);
        DataType output(DOUBLE, GPU);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_validate, false,false)

        context_manager.SetOperationContext(gpu_context_sync);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output, false,false)

        context_manager.SetOperationContext(default_context);

        auto data = (double *) output.GetData();
        auto data_validate = (double *) output_validate.GetData();
        REQUIRE(output.GetNRow() == size);
        REQUIRE(output.GetNCol() == size);

        double error_threshold = 0.001;
        for (int i = 0; i < output.GetSize(); ++i) {
            double val = fabs(data[i] - data_validate[i]) / data_validate[i];
            REQUIRE(val <= error_threshold);
        }
        mpcr::kernels::ContextManager::DestroyInstance();
    }
    SECTION("Test Asynchronous and Synchronous GPU contexts") {

        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        auto &context_manager = mpcr::kernels::ContextManager::GetInstance();
        auto default_context = mpcr::kernels::ContextManager::GetOperationContext();

        auto gpu_context_async = context_manager.CreateRunContext("ASYNC");
        gpu_context_async->SetOperationPlacement(GPU);
        gpu_context_async->SetRunMode( RunMode::ASYNC);

        auto gpu_context_sync = context_manager.CreateRunContext("SYNC");
        gpu_context_sync->SetOperationPlacement(GPU);
        gpu_context_sync->SetRunMode( RunMode::SYNC);

        const int size = 1024;
        std::vector<double> values(size * size, 1.0);

        DataType a(values, DOUBLE, GPU);
        a.ToMatrix(size, size);

        DataType b(values, DOUBLE, GPU);
        b.ToMatrix(size, size);

        DataType output_validate_Gemm(DOUBLE, GPU);
        DataType output_validate_Trmm(DOUBLE, GPU);
        DataType output_Gemm(DOUBLE, GPU);
        DataType output_Trmm(DOUBLE, GPU);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_validate_Gemm, false,false)
        SIMPLE_DISPATCH(DOUBLE, linear::Trmm, a, b, output_validate_Trmm, false, true, true, 1)

        context_manager.SetOperationContext(gpu_context_async);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_Gemm, false,false)

        context_manager.SetOperationContext(gpu_context_sync);

        SIMPLE_DISPATCH(DOUBLE, linear::Trmm, a, b, output_Trmm, false, true, true, 1)

        context_manager.SetOperationContext(default_context);

        double error_threshold = 0.001;

        context_manager.SyncContext("ASYNC");

        auto data_Gemm = (double *) output_Gemm.GetData();
        auto data_validate_Gemm = (double *) output_validate_Gemm.GetData();
        REQUIRE(output_Gemm.GetNRow() == size);
        REQUIRE(output_Gemm.GetNCol() == size);

        for (int i = 0; i < output_Gemm.GetSize(); ++i) {
            double val = fabs(data_Gemm[i] - data_validate_Gemm[i]) / data_validate_Gemm[i];
            REQUIRE(val <= error_threshold);
        }

        auto data_Trmm = (double *) output_Trmm.GetData();
        auto data_validate_Trmm = (double *) output_validate_Trmm.GetData();
        REQUIRE(output_Trmm.GetNRow() == size);
        REQUIRE(output_Trmm.GetNCol() == size);

        for (int i = 0; i < output_Trmm.GetSize(); ++i) {
            double val = fabs(data_Trmm[i] - data_validate_Trmm[i]) / data_validate_Trmm[i];
            REQUIRE(val <= error_threshold);
        }
        mpcr::kernels::ContextManager::DestroyInstance();

    }SECTION("Test Concurrent Execution with Multiple Streams (SyncAll)") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(GPU);

        auto& context_manager = mpcr::kernels::ContextManager::GetInstance();
        auto default_context = mpcr::kernels::ContextManager::GetOperationContext();
        auto gpu_context_async1 = context_manager.CreateRunContext("ASYNC1");
        auto gpu_context_async2 = context_manager.CreateRunContext("ASYNC2");

        const int size = 1024;
        std::vector<double> values(size * size, 1.0);

        DataType a(values, DOUBLE, GPU);
        a.ToMatrix(size, size);
        DataType b(values, DOUBLE, GPU);
        b.ToMatrix(size, size);
        DataType output1(DOUBLE, GPU);
        DataType output2(DOUBLE, GPU);
        DataType output_Validate1(DOUBLE, GPU);
        DataType output_Validate2(DOUBLE, GPU);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_Validate1, false, false)
        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_Validate2, false, false)

        gpu_context_async1->SetOperationPlacement(GPU);
        gpu_context_async2->SetOperationPlacement(GPU);

        gpu_context_async1->SetRunMode(RunMode::ASYNC);
        gpu_context_async2->SetRunMode(RunMode::ASYNC);

        context_manager.SetOperationContext(gpu_context_async1);
        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output1, false, false)
        context_manager.SetOperationContext(gpu_context_async2);
        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output2, false, false)

        context_manager.SyncAll();

        double error_threshold = 0.001;

        context_manager.SetOperationContext(default_context);
        auto data1 = (double *) output1.GetData();
        auto data_validate1 = (double *) output_Validate1.GetData();
        REQUIRE(output1.GetNRow() == size);
        REQUIRE(output1.GetNCol() == size);

        for (int i = 0; i < output1.GetSize(); ++i) {
            double val = fabs(data1[i] - data_validate1[i]) / data_validate1[i];
            REQUIRE(val <= error_threshold);
        }

        auto data2 = (double *) output2.GetData();
        auto data_validate2 = (double *) output_Validate2.GetData();
        REQUIRE(output2.GetNRow() == size);
        REQUIRE(output2.GetNCol() == size);

        for (int i = 0; i < output2.GetSize(); ++i) {
            double val = fabs(data2[i] - data_validate2[i]) / data_validate2[i];
            REQUIRE(val <= error_threshold);
        }

        mpcr::kernels::ContextManager::DestroyInstance();
    }
    SECTION("Test Async-to-Sync Transition on the Same Stream") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        auto &context_manager = mpcr::kernels::ContextManager::GetInstance();
        auto default_context = mpcr::kernels::ContextManager::GetOperationContext();

        auto gpu_context_async = context_manager.CreateRunContext("GPU");
        gpu_context_async->SetOperationPlacement(GPU);
        gpu_context_async->SetRunMode(RunMode::ASYNC);

        const int size = 1024;
        std::vector<double> values(size * size, 1.0);

        DataType a(values, DOUBLE, GPU);
        a.ToMatrix(size, size);

        DataType b(values, DOUBLE, GPU);
        b.ToMatrix(size, size);

        DataType output_validate(DOUBLE, GPU);
        DataType output(DOUBLE, GPU);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_validate, false, false)

        context_manager.SetOperationContext(gpu_context_async);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output, false, false)

        gpu_context_async->SetRunMode(RunMode::SYNC);
        context_manager.SetOperationContext(default_context);

        auto data = (double *) output.GetData();
        auto data_validate = (double *) output_validate.GetData();
        REQUIRE(output.GetNRow() == size);
        REQUIRE(output.GetNCol() == size);

        double error_threshold = 0.001;
        for (int i = 0; i < output.GetSize(); ++i) {
            double val = fabs(data[i] - data_validate[i]) / data_validate[i];
            REQUIRE(val <= error_threshold);
        }
        mpcr::kernels::ContextManager::DestroyInstance();
    }
}
TEST_CASE("Cuda Streams", "[Cuda Streams]") {
#ifdef USE_CUDA
    TEST_CUDA_STREAMS();
#endif
}