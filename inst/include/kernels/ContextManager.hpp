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
            SetOperationContext(RunContext *&aRunContext);

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

            /**
             * @brief
             * Delete a specific RunContext from the context manager.
             */
            void
            DeleteRunContext(size_t aIdx);

            /**
             * @brief
             * Get a GPU context to be used internally if any stream is needed.
             */
            static
            RunContext *
            GetGPUContext();


        protected:
            /**
             * @brief
             * Context Manager constructor.
             *
             */
            ContextManager()=default;


        private:
            /** map holding all the created run context **/
            std::map<int ,mpcr::kernels::RunContext *> mContexts;
            /** Pointer to hold the current run context to be used internally
             *  if any streams is needed.
             **/
            RunContext *mpCurrentContext;
#ifdef USE_CUDA
            /** Context used internally if any GPU context is needed, when the
             *  Current context is CPU
             **/
             RunContext *mpGPUContext;
#endif
        };


    }
}


#endif //MPCR_CONTEXTMANAGER_HPP
