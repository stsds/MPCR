/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <kernels/ContextManager.hpp>


using namespace mpcr::kernels;
using namespace mpcr::definitions;

ContextManager *ContextManager::mpInstance = nullptr;


ContextManager &
ContextManager::GetInstance() {
    if (mpInstance == nullptr) {
        mpInstance = new ContextManager();
        mpInstance->mContexts["default"] = new RunContext();
        mpInstance->mpCurrentContext = mpInstance->mContexts["default"];
#ifdef USE_CUDA
        mpInstance->mpGPUContext = new RunContext(definitions::GPU,
                                                  RunMode::SYNC);
#endif

    }
    return *mpInstance;
}

void
ContextManager::SyncContext(const std::string &aRunContextName) const {

    auto it = mContexts.find(aRunContextName);
    if (it == mContexts.end()) {
        MPCR_API_EXCEPTION("No stream with that name", -1);
    }
    mContexts.at(aRunContextName)->Sync();

}


void
ContextManager::SyncMainContext() const {
    mContexts.at("default")->Sync();
}


void
ContextManager::SyncAll() const {
    for (const auto &[key, context]: mContexts) {
        if (context != nullptr) {
            context->Sync();
        }
    }

}


size_t
ContextManager::GetNumOfContexts() const {
    return mContexts.size();
}


void
ContextManager::DestroyInstance() {
    if (mpInstance) {
        mpInstance->SyncAll();

        for (auto &[key, context]: mpInstance->mContexts) {
            delete context;
            context = nullptr;
        }
        mpInstance->mContexts.clear();

#ifdef USE_CUDA
        delete mpInstance->mpGPUContext;
        mpInstance->mpGPUContext = nullptr;
#endif

        delete mpInstance;
        mpInstance = nullptr;
    }

}


RunContext *
ContextManager::GetContext(const std::string &aRunContextName) {
    auto it = mContexts.find(aRunContextName);
    if (it == mContexts.end()) {
        MPCR_API_EXCEPTION("No stream with that name", -1);
    }
    return mContexts[aRunContextName];
}


void
ContextManager::SetOperationContext(RunContext *&aRunContext) {
    this->mpCurrentContext = aRunContext;
}


RunContext *
ContextManager::GetOperationContext() {
    if (mpInstance == nullptr) {
        ContextManager::GetInstance();
    }
    if (mpInstance->mpCurrentContext == nullptr) {
        MPCR_API_EXCEPTION("No current operation context available", -1);
    }
    return mpInstance->mpCurrentContext;
}


RunContext *
ContextManager::CreateRunContext(const std::string &aRunContextName,
                                 const OperationPlacement &aOperationPlacement,
                                 const RunMode  &aRunMode) {
    auto it = mContexts.find(aRunContextName);
    if (it != mContexts.end()) {
        MPCR_API_EXCEPTION("A stream with that name already exists", -1);
    }
    auto run_context = new mpcr::kernels::RunContext();
    mpInstance->mContexts[aRunContextName] = run_context;
    run_context->SetOperationPlacement(aOperationPlacement);
    run_context->SetRunMode(aRunMode);
    return run_context;
}


void ContextManager::DeleteRunContext(const std::string &aRunContextName) {
    if (aRunContextName == "default") {
        MPCR_API_WARN("Cannot delete default RunContext", 1);
        return;
    }
    auto it = mContexts.find(aRunContextName);
    if (it == mContexts.end()) {
        MPCR_API_EXCEPTION("No stream with that name", -1);
        return;
    }
    if (this->mpCurrentContext == this->GetContext(aRunContextName)) {
        auto default_context = this->GetContext("default");
        this->SetOperationContext(default_context);
    }
    auto deleted_context = this->GetContext(aRunContextName);
    if (!deleted_context) {
        MPCR_API_EXCEPTION("Failed to retrieve context", -1);
        return;
    }
    auto erase = mpInstance->mContexts.find(aRunContextName);
    if (erase != mpInstance->mContexts.end()) {
        mpInstance->mContexts.erase(erase);
    }
    delete deleted_context;
    deleted_context = nullptr;
}

RunContext *
ContextManager::GetGPUContext() {
#ifdef USE_CUDA
    if (mpInstance == nullptr) {
        ContextManager::GetInstance();
    }
    if (mpInstance->mpGPUContext == nullptr) {
        MPCR_API_EXCEPTION("No current operation context available", -1);
    }
    return mpInstance->mpGPUContext;
#else
    MPCR_API_EXCEPTION("Code is compiled without CUDA support", -1);
#endif
    return nullptr;
}

std::vector<std::string>
ContextManager::GetAllContextNames() const {
    std::vector<std::string> contextNames;
    for (const auto &[key, _] : mContexts) {
        contextNames.push_back(key);
    }
    return contextNames;
}

void
ContextManager::FinalizeRunContext(const std::string &aRunContextName){
    auto &ContextManager = mpcr::kernels::ContextManager::GetInstance();
    auto context = ContextManager.GetContext(aRunContextName);
    context->FinalizeRunContext();
}