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
        mpInstance->mContexts[ 0 ] = new RunContext();
        mpInstance->mpCurrentContext = mpInstance->mContexts[ 0 ];
#ifdef USE_CUDA
        mpInstance->mpGPUContext = new RunContext(definitions::GPU,
                                                  RunMode::SYNC);
#endif

    }
    return *mpInstance;
}


void
ContextManager::SyncContext(size_t aIdx) const {
    if (aIdx >= mContexts.size()) {
        MPCR_API_EXCEPTION("Trying to fetch invalid Context Idx", -1);
    }

    mContexts.at(aIdx)->Sync();

}


void
ContextManager::SyncMainContext() const {
    mContexts.at(0)->Sync();
}


void
ContextManager::SyncAll() const {
    for (const auto& [key, context] : mContexts) {
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

        for (auto& [key, context] : mpInstance->mContexts) {
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
ContextManager::GetContext(size_t aIdx) {
    if (aIdx >= mContexts.size()) {
        MPCR_API_EXCEPTION("Trying to fetch invalid Context Idx", -1);
        return nullptr;
    }else{
        return mContexts[ aIdx ];
    }
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
ContextManager::CreateRunContext() {
    auto run_context = new mpcr::kernels::RunContext();
    int newKey = mpInstance->mContexts.empty() ? 0 : mpInstance->mContexts.rbegin()->first + 1;
    mpInstance->mContexts[newKey] = run_context;
    return mpInstance->mContexts[newKey];
}

void
ContextManager::DeleteRunContext(size_t aIdx) {
    auto deleted_context = this->GetContext(aIdx);
    auto current_context = ContextManager::GetOperationContext();
    auto it = mpInstance->mContexts.find(aIdx);
    if (it != mpInstance->mContexts.end()) {
        mpInstance->mContexts.erase(it);
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
#endif
    return nullptr;
}
