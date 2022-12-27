
#include <data-units/Precision.hpp>
#include <libraries/catch/catch.hpp>


using namespace mpr::precision;
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
    REQUIRE(temp == INT);

    temp = GetInputPrecision(2);
    REQUIRE(temp == FLOAT);

    temp = GetInputPrecision(3);
    REQUIRE(temp == DOUBLE);

    temp = GetInputPrecision("int");
    REQUIRE(temp == INT);

    temp = GetInputPrecision("FLOAT");
    REQUIRE(temp == FLOAT);

    temp = GetInputPrecision("doUble");
    REQUIRE(temp == DOUBLE);

    temp = GetInputPrecision(INT);
    REQUIRE(temp == INT);

    temp = GetInputPrecision(FLOAT);
    REQUIRE(temp == FLOAT);

    temp = GetInputPrecision(DOUBLE);
    REQUIRE(temp == DOUBLE);

    /** Testing Output and Operation Generator**/
    Precision a = INT;
    Precision b = INT;
    temp = GetOutputPrecision(a, b);
    REQUIRE(temp == INT);
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
    b = INT;
    temp = GetOutputPrecision(a, b);
    REQUIRE(temp == FLOAT);
    temp = GetOperationPrecision(a, b, temp);
    REQUIRE(temp == FSF);
    temp = FLOAT;
    temp = GetOperationPrecision(b, a, temp);
    REQUIRE(temp == SFF);


    /****************************************************/

    a = INT;
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
