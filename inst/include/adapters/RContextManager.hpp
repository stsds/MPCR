
#ifndef MPCR_RCONTEXTMANAGER_HPP
#define MPCR_RCONTEXTMANAGER_HPP

#include <kernels/ContextManager.hpp>
#include <sstream>
#include <algorithm>

using namespace mpcr::kernels;

void
SetOperationPlacement(const std::string &aOperationPlacement);

std::string
GetOperationPlacement();

std::string
GetRunMode(std::string &aRunContextName);

void
SetRunMode(std::string &aRunContextName, std::string &aRunMode);
void
FinalizeSyncOperations(std::string &aRunContextName);

void
CreateRunContext(std::string &aRunContextName);

ContextManager &
GetInstance();

void
SyncContext(const std::string &aRunContextName);

void
SyncAll();

size_t
GetNumOfContexts();

RunContext *
GetContext(const std::string &aRunContextName);

void
SetOperationContext(std::string &aRunContextName);

void
DeleteRunContext(const std::string &aRunContextName);

#endif //MPCR_RCONTEXTMANAGER_HPP
