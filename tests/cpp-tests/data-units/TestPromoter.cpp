
#include <libraries/catch/catch.hpp>
#include <data-units/Promoter.hpp>


using namespace mpr::precision;


void
TEST_PROMOTE() {
    SECTION("Test Basic Promoter") {
        std::cout << "Testing Promoter ..." << std::endl;
        DataType a(FLOAT);
        DataType b(HALF);
        DataType c(DOUBLE);

        Promoter p(3);
        p.Insert(a);
        p.Insert(b);
        p.Insert(c);

        p.Promote();

        REQUIRE(a.GetPrecision() == DOUBLE);
        REQUIRE(b.GetPrecision() == DOUBLE);
        REQUIRE(c.GetPrecision() == DOUBLE);

        p.DePromote();

        REQUIRE(a.GetPrecision() == DOUBLE);
        REQUIRE(b.GetPrecision() == HALF);
        REQUIRE(c.GetPrecision() == DOUBLE);
    }SECTION("Test Get Precision Mapper") {
        std::cout << "Testing Promoter Precision Mapper ..." << std::endl;
        auto a = new DataType(FLOAT);



        Promoter p(1);
        auto temp_tile = p.GetPromotedTile(a, DOUBLE);
        auto temp_tile_two = p.GetPromotedTile(a, DOUBLE);
        REQUIRE(temp_tile->GetPrecision() == DOUBLE);
        REQUIRE(temp_tile == temp_tile_two);

        temp_tile = p.GetPromotedTile(a, FLOAT);
        REQUIRE(temp_tile->GetPrecision() == FLOAT);
        REQUIRE(temp_tile == a);

        temp_tile = p.GetPromotedTile(a, HALF);
        REQUIRE(temp_tile->GetPrecision() == HALF);

        temp_tile_two = p.GetPromotedTile(a, HALF);
        REQUIRE(temp_tile==temp_tile_two);

        delete a;

    }


}


TEST_CASE("Promoter Test", "[Promoter]") {
    TEST_PROMOTE();
}
