/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <adapters/RContextManager.hpp>
#include <sstream>
#include <algorithm>

using namespace mpcr::kernels;

void
SetOperationPlacement(const std::string &aOperationPlacement, const std::string &aRunContextName) {
    auto operation_placement = mpcr::definitions::GetInputOperationPlacement(
            aOperationPlacement);
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    ContextManager.GetContext(aRunContextName)->SetOperationPlacement(operation_placement);
}


std::string
GetOperationPlacement(const std::string &aRunContextName) {
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    auto operation_placement = ContextManager.GetContext(aRunContextName)->GetOperationPlacement();
    return operation_placement == mpcr::definitions::CPU ? "CPU" : "GPU";
}

void
SetRunMode(std::string &aRunContextName, std::string &aRunMode) {
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    std::transform(aRunMode.begin(), aRunMode.end(),
                   aRunMode.begin(), ::tolower);
    mpcr::kernels::RunMode run_mode;
    if (aRunMode == "async") {
        run_mode = RunMode::ASYNC;
    }else{
        run_mode = RunMode::SYNC;
    }
    ContextManager.GetContext(aRunContextName)->SetRunMode(
            run_mode);
}

std::string
GetRunMode(std::string &aRunContextName) {
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    auto run_mode = ContextManager.GetContext(aRunContextName)->GetRunMode();
    return run_mode == RunMode::SYNC ? "SYNC" : "ASYNC" ;
}

void
FinalizeSyncOperations(std::string &aRunContextName){
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    auto run_mode = ContextManager.GetContext(aRunContextName)->GetRunMode();
    if(run_mode == RunMode::SYNC){
        ContextManager.GetContext(aRunContextName)->Sync();
        ContextManager.GetContext(aRunContextName)->FreeWorkBufferHost();
    }
}

void
CreateRunContext(std::string &aRunContextName){
    auto context = ContextManager::CreateRunContext(aRunContextName);
}

ContextManager &
GetInstance(){
    return ContextManager::GetInstance();
}

void
SyncContext(const std::string &aRunContextName) {
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    ContextManager.SyncContext(aRunContextName);
}

void
SyncAll(){
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    ContextManager.SyncAll();
}

size_t
GetNumOfContexts(){
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    return ContextManager.GetNumOfContexts();
}

RunContext *
GetContext(const std::string &aRunContextName){
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    return ContextManager.GetContext(aRunContextName);
}

void
SetOperationContext(std::string &aRunContextName){
    auto runContext = GetContext(aRunContextName);
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    ContextManager.SetOperationContext(runContext);
}


void
DeleteRunContext(const std::string &aRunContextName){
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    ContextManager.DeleteRunContext(aRunContextName);
}