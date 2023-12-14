/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_CONTEXTMANAGER_HPP
#define MPCR_CONTEXTMANAGER_HPP

#include <kernels/RunContext.hpp>
#include <vector>


namespace mpcr {
    namespace kernels {
        /**
         * @brief
         * Singleton Class responsible for managing and delivering all MPCR Context Objects.
         */

        class ContextManager {
        private:
            /** Singleton instance of MemoryHandler **/
            static ContextManager *mpInstance;
        public:

            /**
             * @brief
             * Get the current singleton instance, and create one if not instantiated.
             */
            static
            ContextManager &GetInstance();

            /**
             * @brief
             * Destructor to allow correct destruction of instances created.
             */
            ~ContextManager() = default;

            /**
             * Singletons should not be cloneable.
             */
            ContextManager(ContextManager &) = delete;

            /**
             * Singletons should not be assignable.
             */
            void
            operator =(const ContextManager &) = delete;

            /**
             * @brief
             * Synchronizes the kernel stream.
             */
            void
            SyncContext(size_t aIdx = 0) const;

            /**
             * @brief
             * Synchronizes the main kernel stream.
             */
            void
            SyncMainContext() const;

            /**
             * @brief
             * Synchronizes all streams.
             */
            void
            SyncAll() const;

            /**
             * @brief
             * Get number of created streams.
             */
            [[nodiscard]]
            size_t
            GetNumOfContexts() const;

            /**
             * @brief
             * Get specific stream.
             */
            RunContext *
            GetContext(size_t aIdx = 0);

            /**
             * @brief
             * destroy the singleton instance.
             */
            static
            void
            DestroyInstance();

            /**
             * @brief
             * Set the current context to be used internally if any stream is needed.
             */
            void
            SetOperationContext(RunContext *aRunContext);

            /**
             * @brief
             * Get the current context to be used internally if any stream is needed.
             */
            static
            RunContext *
            GetOperationContext();

            /**
             * @brief
             * Create new stream and add it to the context manager.
             */
            RunContext *
            CreateRunContext();


        protected:
            /**
             * @brief
             * Context Manager constructor.
             *
             */
            ContextManager() {
                mpInstance->mContexts = std::vector <RunContext *>(1);
                mpInstance->mContexts[ 0 ] = new RunContext();
                mpInstance->mpCurrentContext = mpInstance->mContexts[ 0 ];
            }


        private:
            /** Vector holding all the created run context **/
            std::vector <mpcr::kernels::RunContext *> mContexts;
            /** Pointer to hold the current run context to be used internally
             *  if any streams is needed.
             **/
            RunContext *mpCurrentContext;
        };


    }
}


#endif //MPCR_CONTEXTMANAGER_HPP
