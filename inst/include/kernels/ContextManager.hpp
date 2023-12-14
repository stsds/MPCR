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

            void
            SyncAll() const;

            [[nodiscard]]
            size_t
            GetNumOfContexts() const;

            RunContext *
            GetContext(size_t aIdx = 0);

            /**
             * @brief
             * destroy the singleton instance.
             */
            static
            void
            DestroyInstance();

            void
            SetOperationContext(RunContext *aRunContext);

            static
            RunContext *
            GetOperationContext();

            RunContext *
            CreateRunContext();


        protected:
            /**
             * @brief
             * Context Manager constructor.
             *
             */
            ContextManager() {
                mpInstance->mContexts = std::vector<RunContext*>(1);
                mpInstance->mContexts[0]=new RunContext();
                mpInstance->mpCurrentContext=mpInstance->mContexts[0];
            }

        private:
            std::vector <mpcr::kernels::RunContext*> mContexts;
            RunContext *mpCurrentContext;

        };


    }
}




#endif //MPCR_CONTEXTMANAGER_HPP
