
#ifndef MPCR_RCONTEXTMANAGER_HPP
#define MPCR_RCONTEXTMANAGER_HPP

#include <kernels/ContextManager.hpp>
#include <sstream>
#include <algorithm>

using namespace mpcr::kernels;

/**
 * @brief
 * Set the Operation placement, to indicate whether the context is
 * for GPU or CPU.
 *
 * GPU context can work for both CPU & GPU operations, however,
 * the CPU context can't
 *
 * @param[in] aOperationPlacement
 * Operation placement enum CPU,GPU.
 * @param[in] aRunContextName
 * RunContext name.
 *
 */
void
SetOperationPlacement(std::string &aRunContextName, const std::string &aOperationPlacement);

/**
 * @brief
 * Get the Operation placement, indicating whether the context is
 * for GPU or CPU.
 *
 * GPU context can work for both CPU & GPU operations, however,
 * the CPU context can't
 *
 * @returns
 * Operation placement.
 */
std::string
GetOperationPlacement(std::string &aRunContextName);

/**
 * @brief
 * Get the RunMode for the context, indicating whether the context is
 * SYNC or ASYNC, useful only in the case of GPU
 * @param[in] aRunContextName
 * RunContext name.
 *
 * @returns
 * Run Mode
 */
std::string
GetRunMode(std::string &aRunContextName);

/**
 * @brief
 * Set the RunMode for the context, to indicate whether the context is
 * SYNC or ASYNC, useful only in the case of GPU
 * @param[in] aRunMode
 * Run Mode enum indicating whether the context is SYNC or ASYNC
 */
void
SetRunMode(std::string &aRunContextName, std::string &aRunMode);

/**
 * @brief
 * Cleans up and synchronizes resources.
 * Frees the host work buffer and syncs.
 * @param[in] aRunContextName
 * RunContext name.
 */
void
FinalizeRunContext(std::string &aRunContextName);

/**
 * @brief
 * Create new stream and add it to the context manager.
 * @param[in] aRunContextName
 * RunContext name.
 */
void
CreateRunContext(const std::string &aRunContextName,
                 const std::string &aOperationPlacement = "CPU",
                 const std::string  &aRunMode = "SYNC");

/**
 * @brief
 * Synchronizes the kernel stream.
 * @param[in] aRunContextName
 * RunContext name.
 */
void
SyncContext(const std::string &aRunContextName);

/**
 * @brief
 * Synchronizes all streams.
 */
void
SyncAll();

/**
 * @brief
 * Get number of created streams.
 *
 * @returns
 * Run size_t
 */
size_t
GetNumOfContexts();

/**
 * @brief
 * Set the current context to be used internally if any stream is needed.
 * @param[in] aRunContextName
 * RunContext name.
 */
void
SetOperationContext(std::string &aRunContextName);

/**
 * @brief
 * Delete a specific from the context manager.
 * @param[in] aRunContextName
 * RunContext name.
 */
void
DeleteRunContext(const std::string &aRunContextName);


/**
 * @brief
 * Retrieve the names of all existing RunContext instances managed by the ContextManager.
 *
 * @returns
 * A vector of strings containing the names of all RunContext instances.
 */
std::vector<std::string>
GetAllContextNames();

#endif //MPCR_RCONTEXTMANAGER_HPP
