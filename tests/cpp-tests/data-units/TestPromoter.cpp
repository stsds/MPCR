
#include <libraries/catch/catch.hpp>
#include <data-units/Promoter.hpp>

using namespace mpr::precision;


void
TEST_PROMOTE(){

    std::cout<<"Testing Promoter ..."<<std::endl;
    DataType a(FLOAT);
    DataType b(INT);
    DataType c(DOUBLE);

    Promoter p(3);
    p.Insert(a);
    p.Insert(b);
    p.Insert(c);

    p.Promote();

    REQUIRE(a.GetPrecision()==DOUBLE);
    REQUIRE(b.GetPrecision()==DOUBLE);
    REQUIRE(c.GetPrecision()==DOUBLE);

    p.DePromote();

    REQUIRE(a.GetPrecision()==DOUBLE);
    REQUIRE(b.GetPrecision()==INT);
    REQUIRE(c.GetPrecision()==DOUBLE);


}

TEST_CASE("Promoter Test", "[Promoter]") {
TEST_PROMOTE();
}
