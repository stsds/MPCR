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
        mpInstance->mContexts.resize(1);
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

    mContexts[ aIdx ]->Sync();

}


void
ContextManager::SyncMainContext() const {
    mContexts[ 0 ]->Sync();
}


void
ContextManager::SyncAll() const {
    for (auto &context: mContexts) {
        context->Sync();
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
        for (auto *&x: mpInstance->mContexts) {
            delete x;
            x = nullptr;
        }

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
    }

    return mContexts[ aIdx ];
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
    auto run_context = new RunContext();
    mpInstance->mContexts.push_back(run_context);
    return mpInstance->mContexts.back();
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
