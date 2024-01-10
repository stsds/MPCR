/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/


#include <data-units/DataHolder.hpp>
#include <libraries/catch/catch.hpp>


using namespace mpcr;
using namespace mpcr::kernels;
using namespace mpcr::memory;
using namespace std;


#ifdef USE_CUDA


void
TEST_DATA_HOLDER_GPU() {
    ContextManager::GetOperationContext()->SetOperationPlacement(GPU);
    ContextManager::GetOperationContext()->SetRunMode(RunMode::SYNC);

    auto n = 80;
    SECTION("Creating CPU Buffer") {
        cout<<"Testing Data Holder with GPU support ..."<<endl;
        DataHolder data_a(n * sizeof(float), CPU);
        auto pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = i;
        }


        data_a.SetDataPointer((char *) pdata_host, n * sizeof(float), CPU);
        pdata_host = (float *) data_a.GetDataPointer(CPU);
        REQUIRE(data_a.GetSize() == n * sizeof(float));

        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_host[ i ] == i);
        }

        SECTION("Getting a GPU copy") {
            auto pdata_device = (float *) data_a.GetDataPointer(GPU);

            auto pdata_copy_host = memory::AllocateArray(n * sizeof(float), CPU,
                                                         nullptr);
            memory::MemCpy(pdata_copy_host, (char *) pdata_device,
                           n * sizeof(float),
                           ContextManager::GetOperationContext(),
                           MemoryTransfer::DEVICE_TO_HOST);

            pdata_host = (float *) pdata_copy_host;
            for (auto i = 0; i < n; i++) {
                REQUIRE(pdata_host[ i ] == i);
            }

            memory::DestroyArray(pdata_copy_host, CPU, nullptr);

        }

    }SECTION("Buffer Deletion") {

        DataHolder data_a(n * sizeof(float), CPU);
        auto pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = i;
        }


        data_a.SetDataPointer((char *) pdata_host, n * sizeof(float), CPU);
        auto pdata_device = (float *) data_a.GetDataPointer(GPU);
        data_a.FreeMemory(CPU);
        pdata_host = (float *) data_a.GetDataPointer(CPU);

        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_host[ i ] == i);
        }

        data_a.FreeMemory(CPU);
        REQUIRE(data_a.GetSize() == n * sizeof(float));
        data_a.FreeMemory(GPU);
        REQUIRE(data_a.GetDataPointer(CPU) == nullptr);
        REQUIRE(data_a.GetDataPointer(GPU) == nullptr);
        REQUIRE(data_a.GetSize() == 0);

        data_a.Allocate(n * sizeof(float), CPU);
        REQUIRE(data_a.GetSize() == n * sizeof(float));
        REQUIRE(data_a.GetDataPointer(CPU) != nullptr);

        pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = 2 * i;
        }

        pdata_device = (float *) data_a.GetDataPointer(GPU);
        auto pdata_copy_host = memory::AllocateArray(n * sizeof(float), CPU,
                                                     nullptr);
        memory::MemCpy(pdata_copy_host, (char *) pdata_device,
                       n * sizeof(float),
                       ContextManager::GetOperationContext(),
                       MemoryTransfer::DEVICE_TO_HOST);

        pdata_host = (float *) pdata_copy_host;
        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_host[ i ] == 2 * i);
        }

        pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = i * 3;
        }
        data_a.SetDataPointer((char *) pdata_host, n * sizeof(float), CPU);

        pdata_device = (float *) data_a.GetDataPointer(GPU);
        memory::MemCpy(pdata_copy_host, (char *) pdata_device,
                       n * sizeof(float),
                       ContextManager::GetOperationContext(),
                       MemoryTransfer::DEVICE_TO_HOST);

        pdata_host = (float *) pdata_copy_host;
        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_host[ i ] == 3 * i);
        }

        data_a.ClearUp();
        REQUIRE(data_a.GetSize() == 0);
        REQUIRE(data_a.GetDataPointer(CPU) == nullptr);
        REQUIRE(data_a.GetDataPointer(GPU) == nullptr);

    }SECTION("Changing precision on Host") {
        DataHolder data_a(n * sizeof(float), CPU);
        auto pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = 2 * i;
        }

        data_a.SetDataPointer((char *) pdata_host, n * sizeof(float), CPU);
        data_a.ChangePrecision <float, double>();
        REQUIRE(data_a.GetSize() == n * sizeof(double));
        auto pdata_host_double = (double *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_host_double[ i ] == 2 * i);
        }

        data_a.ChangePrecision <double, double>();
        REQUIRE(data_a.GetSize() == n * sizeof(double));
        REQUIRE(pdata_host_double == (double *) data_a.GetDataPointer(CPU));
    }SECTION("Changing precision on Device") {
        DataHolder data_a(n * sizeof(float), CPU);
        auto pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = 2 * i;
        }

        data_a.SetDataPointer((char *) pdata_host, n * sizeof(float), CPU);
        data_a.GetDataPointer(GPU);
        data_a.FreeMemory(CPU);

        data_a.ChangePrecision <float, double>();
        REQUIRE(data_a.GetSize() == n * sizeof(double));
        auto pdata_host_double = (double *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_host_double[ i ] == 2 * i);
        }

        data_a.ChangePrecision <double, double>();
        REQUIRE(data_a.GetSize() == n * sizeof(double));
        REQUIRE(pdata_host_double == (double *) data_a.GetDataPointer(CPU));
    }SECTION("Changing precision depending on Context") {
        DataHolder data_a(n * sizeof(float), CPU);
        auto pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = 2 * i;
        }

        data_a.SetDataPointer((char *) pdata_host, n * sizeof(float), CPU);
        auto pdata_device = data_a.GetDataPointer(GPU);

        ContextManager::GetOperationContext()->SetOperationPlacement(GPU);
        ContextManager::GetOperationContext()->SetRunMode(RunMode::SYNC);

        data_a.ChangePrecision <float, double>();
        REQUIRE(data_a.GetSize() == n * sizeof(double));
        auto pdata_host_double = (double *) data_a.GetDataPointer(CPU);
        auto pdata_device_new = data_a.GetDataPointer(GPU);

        REQUIRE(pdata_device != pdata_device_new);

        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_host_double[ i ] == 2 * i);
        }

        data_a.ChangePrecision <double, double>();
        REQUIRE(data_a.GetSize() == n * sizeof(double));
        REQUIRE(pdata_host_double == (double *) data_a.GetDataPointer(CPU));
    }


}
#else

void
TEST_DATA_HOLDER_CPU() {
    ContextManager::GetOperationContext()->SetOperationPlacement(CPU);
    ContextManager::GetOperationContext()->SetRunMode(RunMode::SYNC);

    auto n = 80;
    SECTION("Creating CPU Buffer") {
        cout<<"Testing Data Holder without GPU support ..."<<endl;
        DataHolder data_a(n * sizeof(float), CPU);
        auto pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = i;
        }


        data_a.SetDataPointer((char *) pdata_host, n * sizeof(float), CPU);
        pdata_host = (float *) data_a.GetDataPointer(CPU);
        REQUIRE(data_a.GetSize() == n * sizeof(float));

        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_host[ i ] == i);
        }

        SECTION("Getting a GPU copy") {
            REQUIRE_THROWS(data_a.GetDataPointer(GPU));
        }

    }SECTION("Buffer Deletion") {

        DataHolder data_a(n * sizeof(float), CPU);
        auto pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = i;
        }


        data_a.SetDataPointer((char *) pdata_host, n * sizeof(float), CPU);
        data_a.FreeMemory(CPU);
        pdata_host = (float *) data_a.GetDataPointer(CPU);
        REQUIRE(pdata_host== nullptr);
        REQUIRE(data_a.GetSize()==0);


    }SECTION("Changing precision on Host") {
        DataHolder data_a(n * sizeof(float), CPU);
        auto pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = 2 * i;
        }

        data_a.SetDataPointer((char *) pdata_host, n * sizeof(float), CPU);
        REQUIRE_THROWS(data_a.GetDataPointer(GPU));

        data_a.ChangePrecision <float, double>();
        REQUIRE(data_a.GetSize() == n * sizeof(double));
        auto pdata_host_double = (double *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_host_double[ i ] == 2 * i);
        }

        data_a.ChangePrecision <double, double>();
        REQUIRE(data_a.GetSize() == n * sizeof(double));
        REQUIRE(pdata_host_double == (double *) data_a.GetDataPointer(CPU));

        REQUIRE_THROWS(data_a.GetDataPointer(GPU));

    }SECTION("Changing precision depending on Context") {
        DataHolder data_a(n * sizeof(float), CPU);
        auto pdata_host = (float *) data_a.GetDataPointer(CPU);
        for (auto i = 0; i < n; i++) {
            pdata_host[ i ] = 2 * i;
        }

        data_a.SetDataPointer((char *) pdata_host, n * sizeof(float), CPU);
        auto pdata_host_raw = data_a.GetDataPointer(CPU);


        data_a.ChangePrecision <float, double>();
        REQUIRE(data_a.GetSize() == n * sizeof(double));
        auto pdata_host_double = (double *) data_a.GetDataPointer(CPU);
        auto pdata_host_new = data_a.GetDataPointer(CPU);

        REQUIRE(pdata_host_raw != pdata_host_new);

        for (auto i = 0; i < n; i++) {
            REQUIRE(pdata_host_double[ i ] == 2 * i);
        }

        data_a.ChangePrecision <double, double>();
        REQUIRE(data_a.GetSize() == n * sizeof(double));
        REQUIRE(pdata_host_double == (double *) data_a.GetDataPointer(CPU));
    }
}
#endif



TEST_CASE("TestDataHolder", "[DataHolder]") {
#ifdef USE_CUDA
    TEST_DATA_HOLDER_GPU();
#else
    TEST_DATA_HOLDER_CPU();
#endif
}
