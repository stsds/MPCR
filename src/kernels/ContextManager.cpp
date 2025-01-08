/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <kernels/ContextManager.hpp>


using namespace mpcr::kernels;

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


void ContextManager::DestroyInstance() {
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
ContextManager::CreateRunContext(const std::string &aRunContextName) {
    auto run_context = new mpcr::kernels::RunContext();
    mpInstance->mContexts[aRunContextName] = run_context;
    return run_context;
}


void ContextManager::DeleteRunContext(const std::string &aRunContextName) {
    auto it = mContexts.find(aRunContextName);
    if(this->mpCurrentContext == this->GetContext(aRunContextName)){
        auto default_context = this->GetContext("default");
        this->SetOperationContext(default_context);
    }
        if (it == mContexts.end()) {
        MPCR_API_EXCEPTION("No stream with that name", -1);
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
#endifmpCurrentContext
    return nullptr;
#endif USE_CUDA
}