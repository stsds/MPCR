/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <libraries/catch/catch.hpp>
#include <kernels/Promoter.hpp>


using namespace mpcr::precision;
using namespace mpcr::kernels;


void
TEST_PROMOTE() {
    SECTION("Test Basic Promoter") {
        std::cout << "Testing Promoter ..." << std::endl;
        DataType a(FLOAT);
//        DataType b(HALF);
        DataType c(DOUBLE);

        Promoter p(2);
//        Promoter p(3);
        p.Insert(a);
//        p.Insert(b);
        p.Insert(c);

        p.Promote();

        REQUIRE(a.GetPrecision() == DOUBLE);
//        REQUIRE(b.GetPrecision() == DOUBLE);
        REQUIRE(c.GetPrecision() == DOUBLE);

        p.DePromote();

        REQUIRE(a.GetPrecision() == DOUBLE);
//        REQUIRE(b.GetPrecision() == HALF);
        REQUIRE(c.GetPrecision() == DOUBLE);
    }

}


TEST_CASE("Promoter Test", "[Promoter]") {
    TEST_PROMOTE();
}
