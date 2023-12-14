/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <kernels/Precision.hpp>
#include <libraries/catch/catch.hpp>


using namespace mpcr::precision;
using namespace std;


/**
 * Exceptions cannot be tested inside C++ since Rcpp::stop throw segmentation
 * fault if called inside C++ due to stack trace ();
 **/

void
TEST_PRECISION() {
    cout << "Testing Precision ..." << endl;

    /** Testing Input Generator**/
    Precision temp = GetInputPrecision(1);
    REQUIRE(temp == HALF);

    temp = GetInputPrecision(2);
    REQUIRE(temp == FLOAT);

    temp = GetInputPrecision(3);
    REQUIRE(temp == DOUBLE);

    temp = GetInputPrecision("half");
    if (USING_HALF) {
        REQUIRE(temp == HALF);
    } else {
        REQUIRE(temp == FLOAT);
    }

    temp = GetInputPrecision("FLOAT");
    REQUIRE(temp == FLOAT);

    temp = GetInputPrecision("doUble");
    REQUIRE(temp == DOUBLE);

    temp = GetInputPrecision(HALF);
    REQUIRE(temp == HALF);

    temp = GetInputPrecision(FLOAT);
    REQUIRE(temp == FLOAT);

    temp = GetInputPrecision(DOUBLE);
    REQUIRE(temp == DOUBLE);

    /** Testing Output and Operation Generator**/
    Precision a = HALF;
    Precision b = HALF;
    temp = GetOutputPrecision(a, b);
    REQUIRE(temp == HALF);
    temp = GetOperationPrecision(a, b, temp);
    REQUIRE(temp == SSS);

    a = FLOAT;
    b = FLOAT;
    temp = GetOutputPrecision(a, b);
    REQUIRE(temp == FLOAT);
    temp = GetOperationPrecision(a, b, temp);
    REQUIRE(temp == FFF);

    a = DOUBLE;
    b = DOUBLE;
    temp = GetOutputPrecision(a, b);
    REQUIRE(temp == DOUBLE);
    temp = GetOperationPrecision(a, b, temp);
    REQUIRE(temp == DDD);

    /****************************************************/

    a = FLOAT;
    b = HALF;
    temp = GetOutputPrecision(a, b);
    REQUIRE(temp == FLOAT);
    temp = GetOperationPrecision(a, b, temp);
    REQUIRE(temp == FSF);
    temp = FLOAT;
    temp = GetOperationPrecision(b, a, temp);
    REQUIRE(temp == SFF);


    /****************************************************/

    a = HALF;
    b = DOUBLE;
    temp = GetOutputPrecision(a, b);
    REQUIRE(temp == DOUBLE);
    temp = GetOperationPrecision(a, b, temp);
    REQUIRE(temp == SDD);
    temp = DOUBLE;
    temp = GetOperationPrecision(b, a, temp);
    REQUIRE(temp == DSD);

    /****************************************************/

    a = FLOAT;
    b = DOUBLE;
    temp = GetOutputPrecision(a, b);
    REQUIRE(temp == DOUBLE);
    temp = GetOperationPrecision(a, b, temp);
    REQUIRE(temp == FDD);
    temp = DOUBLE;
    temp = GetOperationPrecision(b, a, temp);
    REQUIRE(temp == DFD);


    /****************************************************/



}


TEST_CASE("PRECISIONTEST", "[Precision]") {
    TEST_PRECISION();

}
