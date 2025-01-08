/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/


#include <kernels/ContextManager.hpp>
#include <libraries/catch/catch.hpp>


using namespace mpcr::kernels;
using namespace mpcr::definitions;
using namespace std;


void
TEST_CONTEXT_MANAGER() {
    cout << "Testing Context Manager ..." << endl;
    auto default_context = ContextManager::GetOperationContext();
    REQUIRE(ContextManager::GetInstance().GetNumOfContexts() == 1);
    REQUIRE(ContextManager::GetInstance().GetOperationContext() != nullptr);
    REQUIRE(
        ContextManager::GetInstance().GetOperationContext()->GetOperationPlacement() ==
        CPU);
    REQUIRE(
        ContextManager::GetInstance().GetOperationContext()->GetRunMode() ==
        RunMode::SYNC);
    REQUIRE_THROWS(ContextManager::GetInstance().GetContext("RANDOM"));
    REQUIRE_THROWS(ContextManager::GetInstance().SyncContext("RANDOM"));

    auto temp_context = ContextManager::GetInstance().CreateRunContext(std::string("CPU"));
    REQUIRE(ContextManager::GetInstance().GetNumOfContexts() == 2);
    REQUIRE(temp_context->GetOperationPlacement() == CPU);
    REQUIRE(temp_context->GetRunMode() == mpcr::kernels::RunMode::SYNC);
    temp_context->SetOperationPlacement(GPU);
#ifdef USE_CUDA
    REQUIRE(temp_context->GetOperationPlacement() == GPU);
#else
    REQUIRE(temp_context->GetOperationPlacement()==CPU);
#endif
    ContextManager::GetInstance().SetOperationContext(temp_context);
    REQUIRE(
        ContextManager::GetInstance().GetOperationContext() == temp_context);
    ContextManager::GetInstance().DeleteRunContext("CPU");
    REQUIRE(
            ContextManager::GetInstance().GetOperationContext() == default_context);
    ContextManager::DestroyInstance();
    REQUIRE(ContextManager::GetInstance().GetNumOfContexts() == 1);
    REQUIRE(ContextManager::GetInstance().GetOperationContext() != nullptr);


    auto *p1 = &ContextManager::GetInstance();
    auto *p2 = &ContextManager::GetInstance();
    REQUIRE(p1 == p2);


}


TEST_CASE("ContextManagerTest", "[ContextManager]") {
    TEST_CONTEXT_MANAGER();

}
