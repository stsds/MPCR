


#include <libraries/catch/catch.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <operations/LinearAlgebra.hpp>

using namespace std;
using namespace mpr::precision;
using namespace mpr::operations;

void
TEST_LINEAR_ALGEBRA(){
    SECTION("Test CrossProduct"){
        DataType a(5,10,FLOAT);
        DataType b(10,5,FLOAT);
        DataType output(FLOAT);

        for(auto i=0;i<a.GetSize();i++){
            a.SetVal(i,i+1);
        }


        for(auto i=0;i<b.GetSize();i++){
            b.SetVal(i,i+1);
        }

        SIMPLE_DISPATCH(FLOAT,linear::CrossProduct,a,b,output)
        REQUIRE(output.GetNRow()==5);
        REQUIRE(output.GetNCol()==5);

        output.Print();
    }
}


TEST_CASE("LinearAlgebra", "[Linear Algebra]") {
    TEST_LINEAR_ALGEBRA();
}
