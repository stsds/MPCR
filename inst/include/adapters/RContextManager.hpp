
#ifndef MPCR_RCONTEXTMANAGER_HPP
#define MPCR_RCONTEXTMANAGER_HPP

#include <kernels/ContextManager.hpp>


void
SetOperationPlacement(const std::string &aOperationPlacement) {
    auto operation_placement = mpcr::definitions::GetInputOperationPlacement(
        aOperationPlacement);

    mpcr::kernels::ContextManager::GetOperationContext()->SetOperationPlacement(
        operation_placement);
}


std::string
GetOperationPlacement() {
    auto operation_placement = mpcr::kernels::ContextManager::GetOperationContext()->GetOperationPlacement();
    return operation_placement == CPU ? "CPU" : "GPU";
}


#endif //MPCR_RCONTEXTMANAGER_HPP
