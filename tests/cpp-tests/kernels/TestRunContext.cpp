/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/


#include <kernels/RunContext.hpp>
#include <libraries/catch/catch.hpp>


using namespace mpcr::kernels;
using namespace mpcr::definitions;
using namespace std;


#ifdef USE_CUDA
void
TEST_RUN_CONTEXT_CUDA() {
    cout << "Testing Run Context CUDA ..." << endl;
    RunContext context;
    REQUIRE(context.GetOperationPlacement()==mpcr::definitions::CPU);
    REQUIRE(context.GetRunMode()==mpcr::kernels::RunMode::SYNC);

    context.SetOperationPlacement(GPU);
    REQUIRE(context.GetOperationPlacement()==GPU);

    context.SetRunMode(RunMode::ASYNC);
    REQUIRE(context.GetRunMode()==mpcr::kernels::RunMode::ASYNC);

    auto pwork_buffer=context.RequestWorkBufferDevice(500);
    REQUIRE(pwork_buffer!= nullptr);

    auto pwork_buffer_two=context.RequestWorkBufferDevice(600);
    REQUIRE(pwork_buffer_two!= nullptr);

    pwork_buffer=context.RequestWorkBufferDevice(300);
    REQUIRE(pwork_buffer==pwork_buffer_two);

    context.FreeWorkBufferDevice();
    REQUIRE(context.GetRunMode()==mpcr::kernels::RunMode::ASYNC);

    REQUIRE(context.GetInfoPointer()!= nullptr);

    pwork_buffer=context.RequestWorkBufferDevice(400);
    REQUIRE(pwork_buffer!= nullptr);

    context.FreeWorkBufferDevice();
    pwork_buffer_two=context.RequestWorkBufferDevice(0);
    REQUIRE(pwork_buffer_two== nullptr);




}
#endif


void
TEST_RUN_CONTEXT(){
    cout << "Testing Run Context CPU ..." << endl;
    RunContext context;
    REQUIRE(context.GetOperationPlacement()==mpcr::definitions::CPU);
    REQUIRE(context.GetRunMode()==mpcr::kernels::RunMode::SYNC);

    context.SetOperationPlacement(GPU);
    REQUIRE(context.GetOperationPlacement()==CPU);

    context.SetRunMode(RunMode::ASYNC);
    REQUIRE(context.GetRunMode()==mpcr::kernels::RunMode::ASYNC);

    auto context_copy=context;
    REQUIRE(context_copy.GetOperationPlacement()==CPU);
    REQUIRE(context_copy.GetRunMode()==mpcr::kernels::RunMode::ASYNC);

    RunContext context_copy_two(context);
    REQUIRE(context_copy_two.GetOperationPlacement()==CPU);
    REQUIRE(context_copy_two.GetRunMode()==mpcr::kernels::RunMode::ASYNC);

    RunContext context_test_placement(GPU,RunMode::ASYNC);
    REQUIRE(context_test_placement.GetOperationPlacement()==CPU);
    REQUIRE(context_test_placement.GetRunMode()==mpcr::kernels::RunMode::ASYNC);


}

TEST_CASE("RunContextTest", "[RunContext]") {
#ifdef USE_CUDA
    TEST_RUN_CONTEXT_CUDA();
#else
    TEST_RUN_CONTEXT();
#endif

}
