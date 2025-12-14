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

using namespace mpcr::definitions;

void
SetOperationPlacement(std::string &aRunContextName, const std::string &aOperationPlacement) {
    auto operation_placement = mpcr::definitions::GetInputOperationPlacement(
            aOperationPlacement);
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    ContextManager.GetContext(aRunContextName)->SetOperationPlacement(
            operation_placement);
}


std::string
GetOperationPlacement(std::string &aRunContextName) {
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    auto operation_placement = ContextManager.GetContext(aRunContextName)->GetOperationPlacement();
    return operation_placement == mpcr::definitions::CPU ? "CPU" : "GPU";
}

void
SetRunMode(std::string &aRunContextName, std::string &aRunMode) {
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    std::transform(aRunMode.begin(), aRunMode.end(),
                   aRunMode.begin(), ::tolower);
    auto run_mode = mpcr::definitions::GetInputRunMode(
            aRunMode);
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
FinalizeRunContext(std::string &aRunContextName){
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    SyncContext(aRunContextName);
#ifdef USE_CUDA
    ContextManager.GetContext(aRunContextName)->FreeWorkBufferHost();
#endif
}

void
CreateRunContext(std::string &aRunContextName){
    ContextManager::CreateRunContext(aRunContextName);
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

void
SetOperationContext(std::string &aRunContextName){
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    auto runContext = ContextManager.GetContext(aRunContextName);
    ContextManager.SetOperationContext(runContext);
}


void
DeleteRunContext(const std::string &aRunContextName){
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    ContextManager.DeleteRunContext(aRunContextName);
}

std::vector<std::string>
GetAllContextNames(){
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    return ContextManager.GetAllContextNames();
}