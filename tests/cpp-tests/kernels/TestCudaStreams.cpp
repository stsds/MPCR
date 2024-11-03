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
#include "operations/MathematicalOperations.hpp"


using namespace mpcr::operations;
using namespace mpcr::precision;
using namespace mpcr::kernels;
using namespace std;


void
TEST_CUDA_STREAMS() {

    SECTION("Test Synchronous GPU context") {

        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        const int size = 1024;
        std::vector<double> values(size * size, 1.0); // Fill matrix with 1s for simplicity

        DataType a(values, DOUBLE);
        a.ToMatrix(size, size);

        DataType b(values, DOUBLE);
        b.ToMatrix(size, size);

        DataType output_validate(DOUBLE);
        DataType output(DOUBLE);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_validate, false,false)

        auto newContext = mpcr::kernels::ContextManager::GetInstance().CreateRunContext();
        newContext->SetOperationPlacement(GPU);  // Set the new context to use GPU
        newContext->SetRunMode( RunMode::SYNC);   // Set to synchronous mode

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output, false,false)

        REQUIRE(output.GetNRow() == size);
        REQUIRE(output.GetNCol() == size);

        double error_threshold = 0.001;

        for (int i = 0; i < output.GetSize(); ++i) {
            double val = fabs(output.GetVal(i) - output_validate.GetVal(i)) / output_validate.GetVal(i);
            REQUIRE(val <= error_threshold);
        }
        mpcr::kernels::ContextManager::DestroyInstance();
    }

    SECTION("Test Asynchronous GPU context") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        const int size = 1024;
        std::vector<double> values(size * size, 1.0); // Fill matrix with 1s for simplicity

        DataType a(values, DOUBLE);
        a.ToMatrix(size, size);

        DataType b(values, DOUBLE);
        b.ToMatrix(size, size);

        DataType output_validate(DOUBLE);
        DataType output(DOUBLE);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_validate, false, false)

        auto newContext = mpcr::kernels::ContextManager::GetInstance().CreateRunContext();
        newContext->SetOperationPlacement(GPU);  // Set the new context to use GPU
        newContext->SetRunMode(RunMode::ASYNC);   // Set to synchronous mode

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output, false, false)

        REQUIRE(output.GetNRow() == size);
        REQUIRE(output.GetNCol() == size);

        double error_threshold = 0.001;

        for (int i = 0; i < output.GetSize(); ++i) {
            double val = fabs(output.GetVal(i) - output_validate.GetVal(i)) / output_validate.GetVal(i);
            REQUIRE(val <= error_threshold);
        }
        mpcr::kernels::ContextManager::DestroyInstance();
    }

    SECTION("Test Synchronous and Asynchronous GPU contexts") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
                GPU);

        const int size = 1024;
        std::vector<double> values(size * size, 1.0); // Fill matrix with 1s for simplicity

        DataType a(values, DOUBLE);
        a.ToMatrix(size, size);

        DataType b(values, DOUBLE);
        b.ToMatrix(size, size);

        DataType output_validate(DOUBLE);
        DataType outputSync(DOUBLE);
        DataType outputAsync(DOUBLE);

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output_validate, false, false)

        auto& context = mpcr::kernels::ContextManager::GetInstance();
        auto newAyncContext = context.CreateRunContext();
        newAyncContext->SetOperationPlacement(GPU);  // Set the new context to use GPU
        newAyncContext->SetRunMode(RunMode::ASYNC);   // Set to synchronous mode

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, outputAsync, false, false)

        auto newSyncContext = context.CreateRunContext();
        newSyncContext->SetOperationPlacement(GPU);  // Set the new context to use GPU
        newSyncContext->SetRunMode( RunMode::SYNC);   // Set to synchronous mode

        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, outputSync, false,false)

        REQUIRE(outputSync.GetNRow() == size);
        REQUIRE(outputSync.GetNCol() == size);

        double error_threshold = 0.001;

        for (int i = 0; i < outputSync.GetSize(); ++i) {
            double val = fabs(outputSync.GetVal(i) - output_validate.GetVal(i)) / output_validate.GetVal(i);
            REQUIRE(val <= error_threshold);
        }
        std::cout << context.GetNumOfContexts() << std::endl;
        context.SyncContext(1);
        for (int i = 0; i < outputAsync.GetSize(); ++i) {
            double val = fabs(outputAsync.GetVal(i) - output_validate.GetVal(i)) / output_validate.GetVal(i);
            REQUIRE(val <= error_threshold);
        }

        REQUIRE(outputAsync.GetNRow() == size);
        REQUIRE(outputAsync.GetNCol() == size);
        mpcr::kernels::ContextManager::DestroyInstance();


    }SECTION("Test Concurrent Execution with Multiple Streams") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(GPU);

        const int size = 10240;
        std::vector<double> values(size * size, 1.0);

        DataType a(values, DOUBLE);
        a.ToMatrix(size, size);
        DataType b(values, DOUBLE);
        b.ToMatrix(size, size);
        DataType output1(DOUBLE);
        DataType output2(DOUBLE);

        // Create two separate CUDA stream contexts
        auto& context = mpcr::kernels::ContextManager::GetInstance();
        auto streamContext1 = context.CreateRunContext();
        auto streamContext2 = context.CreateRunContext();

        streamContext1->SetOperationPlacement(GPU);
        streamContext2->SetOperationPlacement(GPU);

        streamContext1->SetRunMode(RunMode::ASYNC);
        streamContext2->SetRunMode(RunMode::ASYNC);

        // Launch operations on both streams
        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output1, false, false)
        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, output2, false, false)

        // Wait for both streams to complete
        context.SyncContext(1);
        context.SyncContext(2);

        // Validate results
        REQUIRE(output1.GetNRow() == size);
        REQUIRE(output1.GetNCol() == size);
        REQUIRE(output2.GetNRow() == size);
        REQUIRE(output2.GetNCol() == size);

        mpcr::kernels::ContextManager::DestroyInstance();
    }
    SECTION("Test Async-to-Sync Transition on the Same Stream") {
        mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(GPU);

        const int size = 10240;
        std::vector<double> values(size * size, 1.0);

        DataType a(values, DOUBLE);
        a.ToMatrix(size, size);
        DataType b(values, DOUBLE);
        b.ToMatrix(size, size);
        DataType outputAsync(DOUBLE);
        DataType outputSync(DOUBLE);

        auto& context = mpcr::kernels::ContextManager::GetInstance();
        auto streamContext = context.CreateRunContext();
        streamContext->SetOperationPlacement(GPU);

        // First, set the context to asynchronous and perform an operation
        streamContext->SetRunMode(RunMode::ASYNC);
        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, outputAsync, false, false)

        // Synchronize the asynchronous operation to ensure completion
        context.SyncContext(1);

        // Change the context to synchronous and perform another operation
        streamContext->SetRunMode(RunMode::SYNC);
        SIMPLE_DISPATCH(DOUBLE, linear::CrossProduct, a, b, outputSync, false, false)

        // Validate results
        REQUIRE(outputAsync.GetNRow() == size);
        REQUIRE(outputAsync.GetNCol() == size);
        REQUIRE(outputSync.GetNRow() == size);
        REQUIRE(outputSync.GetNCol() == size);

        mpcr::kernels::ContextManager::DestroyInstance();
    }


}
TEST_CASE("Cuda Streams", "[Cuda Streams]") {
#ifdef USE_CUDA
    TEST_CUDA_STREAMS();
#endif
}