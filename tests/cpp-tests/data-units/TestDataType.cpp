
#include <libraries/catch/catch.hpp>
#include <data-units/DataType.hpp>



void
TEST_DATA_TYPE(){

    DataType array_temp(50,1.5);
    float * data=array_temp.GetData();
    size_t size=array_temp.GetSize();
    for(auto i=0;i<size;i++){
        REQUIRE(data[i]==1.5);
    }

}

TEST_CASE("DataTypeTest", "[DataType]") {
    TEST_DATA_TYPE();
}
