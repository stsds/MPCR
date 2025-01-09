/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <kernels/MemoryHandler.hpp>
#include <libraries/catch/catch.hpp>


using namespace mpcr;
using namespace mpcr::definitions;
using namespace std;


#ifdef USE_CUDA


void
TEST_MEMORY_HANDLER_CUDA() {
    auto n = 80;
    auto pdata_host = new float[n];
    mpcr::kernels::RunContext context;

    for (auto i = 0; i < n; i++) {
        pdata_host[ i ] = i;
    }

    SECTION("Device Allocation") {
        auto *pdata_alloc_device = memory::AllocateArray(sizeof(float) * n, GPU,
                                                         &context);
        REQUIRE(pdata_alloc_device != nullptr);

        REQUIRE_THROWS(memory::MemCpy(pdata_alloc_device, (char *) pdata_host,
                                      sizeof(float) * n, &context,
                                      memory::MemoryTransfer::HOST_TO_DEVICE));

        context.SetOperationPlacement(GPU);
        context.SetRunMode(RunMode::SYNC);

        memory::MemCpy(pdata_alloc_device, (char *) pdata_host,
                       sizeof(float) * n, &context,
                       memory::MemoryTransfer::HOST_TO_DEVICE);

        auto *pdata_alloc_host = memory::AllocateArray(sizeof(float) * n, CPU,
                                                       &context);
        REQUIRE(pdata_alloc_host != nullptr);

        memory::Memset(pdata_alloc_host, 0, sizeof(float) * n, CPU, &context);

        auto *pdata_check_host = (float *) pdata_alloc_host;
        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_check_host[ i ] == 0);
        }

        memory::MemCpy(pdata_alloc_host, pdata_alloc_device, sizeof(float) * n,
                       &context,
                       memory::MemoryTransfer::DEVICE_TO_HOST);

        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_check_host[ i ] == i);
        }

        context.SetOperationPlacement(CPU);
        REQUIRE_THROWS(
            memory::Memset(pdata_alloc_device, 'c', n, GPU, &context));

        context.SetOperationPlacement(GPU);
        memory::Memset(pdata_alloc_device, 'c', n, GPU, &context);

        auto *pdata_alloc_device_two = memory::AllocateArray(sizeof(float) * n,
                                                             GPU, &context);
        REQUIRE(pdata_alloc_device_two != nullptr);

        memory::MemCpy(pdata_alloc_device_two, pdata_alloc_device,
                       sizeof(float) * n, &context,
                       memory::MemoryTransfer::DEVICE_TO_DEVICE);

        for (auto i = 0; i < n; i++) {
            pdata_check_host[ i ] = 0;
        }

        memory::MemCpy(pdata_alloc_host, pdata_alloc_device_two,
                       sizeof(float) * n, &context,
                       memory::MemoryTransfer::DEVICE_TO_HOST);

        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_alloc_host[ i ] == 'c');
        }

        auto idx = n / sizeof(float);
        for (auto i = idx; i < n; i++) {
            REQUIRE(pdata_check_host[ i ] == i);
        }

        memory::DestroyArray(pdata_alloc_device, GPU, &context);
        REQUIRE(pdata_alloc_device == nullptr);

        memory::DestroyArray(pdata_alloc_device_two, GPU, &context);
        REQUIRE(pdata_alloc_device_two == nullptr);

        REQUIRE_THROWS(memory::DestroyArray(pdata_alloc_host, GPU, &context));
        memory::DestroyArray(pdata_alloc_host, CPU, &context);
        REQUIRE(pdata_alloc_host == nullptr);
    }

    delete[] pdata_host;
    SECTION("Test Copy") {

        auto size = 50;

        auto context = kernels::ContextManager::GetGPUContext();
        auto buffer = (int64_t *) memory::AllocateArray(size * sizeof(int64_t),
                                                       GPU, context);

        auto host_buffer = (int64_t *) memory::AllocateArray(
            size * sizeof(int64_t), CPU,
            nullptr);


        for (auto i = 0; i < size; i++) {
            host_buffer[ i ] = i * 2;
        }

        memory::MemCpy((char *) buffer, (char *) host_buffer,
                       sizeof(int64_t) * size, context,
                       memory::MemoryTransfer::HOST_TO_DEVICE);


        auto float_buffer = (float *) memory::AllocateArray(
            size * sizeof(float), GPU, context);


        memory::Copy <int64_t, float>((char *) buffer, (char *) float_buffer,
                                     size, GPU);


        auto host_buffer_val = (float *) memory::AllocateArray(
            size * sizeof(float), CPU, nullptr);

        memory::MemCpy((char *) host_buffer_val, (char *) float_buffer,
                       sizeof(float) * size,
                       context, memory::MemoryTransfer::DEVICE_TO_HOST);

        for (auto i = 0; i < size; i++) {
            REQUIRE(host_buffer_val[ i ] == i * 2);
        }

        auto char_buff = (char *) buffer;
        auto char_float_buff = (char *) float_buffer;
        auto host_one = (char *) host_buffer_val;
        auto host_two = (char *) host_buffer;

        memory::DestroyArray(char_buff, GPU, context);
        memory::DestroyArray(host_one, CPU, context);
        memory::DestroyArray(host_two, CPU, context);
        memory::DestroyArray(char_float_buff, GPU, context);

    }

}


#endif


void
TEST_MEMORY_HANDLER() {
    auto n = 80;
    auto pdata_host = new float[n];
    mpcr::kernels::RunContext context;

    for (auto i = 0; i < n; i++) {
        pdata_host[ i ] = i;
    }

    SECTION("Host Allocation") {
        cout << "Testing Memory Handler ..." << endl;
        context.SetOperationPlacement(CPU);
        auto pdata_alloc_host = memory::AllocateArray(sizeof(float) * n, CPU,
                                                      &context);
        REQUIRE(pdata_alloc_host != nullptr);

        memory::MemCpy(pdata_alloc_host, (char *) pdata_host, sizeof(float) * n,
                       &context,
                       memory::MemoryTransfer::HOST_TO_HOST);


        auto *pdata_alloc_temp = (float *) pdata_alloc_host;

        for (auto i = 0; i < n; i++) {
            REQUIRE((float) pdata_alloc_temp[ i ] == i);
        }

        memory::Memset(pdata_alloc_host, 'c', n, CPU, &context);
        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_alloc_host[ i ] == 'c');
        }

#ifdef USE_CUDA
        REQUIRE_THROWS(memory::DestroyArray(pdata_alloc_host, GPU, &context));
#endif
        memory::DestroyArray(pdata_alloc_host, CPU, &context);
        REQUIRE(pdata_alloc_host == nullptr);

    }

#ifdef USE_CUDA
    TEST_MEMORY_HANDLER_CUDA();
#endif

}


TEST_CASE("MemoryHandlerTest", "[MemoryHandler]") {
    TEST_MEMORY_HANDLER();
}