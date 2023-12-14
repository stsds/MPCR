/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <kernels/ContextManager.hpp>



using namespace mpcr::kernels;

ContextManager* ContextManager::mpInstance= nullptr;

ContextManager &
mpcr::kernels::ContextManager::GetInstance() {
    if(mpInstance == nullptr) {
        mpInstance = new ContextManager();
    }

    return *mpInstance;
}

void
ContextManager::SyncContext(size_t aIdx) const {
    if(aIdx >= mContexts.size()) {
        MPCR_API_EXCEPTION("Trying to fetch invalid Context Idx",-1);
    }
#ifdef USE_CUDA
    mContexts[aIdx]->Sync();
#endif

}


void
ContextManager::SyncMainContext() const {
#ifdef USE_CUDA
    mContexts[0]->Sync();
#endif
}


void
ContextManager::SyncAll() const {
#ifdef USE_CUDA
    for(auto& context : mContexts) {
        context->Sync();
    }
#endif
}


size_t
ContextManager::GetNumOfContexts() const {
    return mContexts.size();
}


void
ContextManager::DestroyInstance() {
    if(mpInstance) {
        mpInstance->SyncAll();
        for(auto &x:mpInstance->mContexts){
            delete x;
            x= nullptr;
        }
        delete mpInstance;
        mpInstance = nullptr;
    }

}


RunContext *
ContextManager::GetContext(size_t aIdx) {
    if(aIdx >= mContexts.size()) {
        MPCR_API_EXCEPTION("Trying to fetch invalid Context Idx",-1);
    }

    return mContexts[aIdx];
}


void
ContextManager::SetOperationContext(RunContext *aRunContext) {
        this->mpCurrentContext=aRunContext;
}


RunContext *
ContextManager::GetOperationContext() {
    if(mpInstance->mpCurrentContext== nullptr){
        MPCR_API_EXCEPTION("No current operation context available",-1);
    }
    return mpInstance->mpCurrentContext;
}


RunContext *
ContextManager::CreateRunContext() {
    auto run_context=new RunContext();
    mpInstance->mContexts.push_back(run_context);
    return run_context;
}


