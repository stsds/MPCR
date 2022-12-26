
#include <libraries/catch/catch.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <data-units/DataType.hpp>


using namespace mpr::precision;

template <typename T>
void
TestSimpleDispatch(T aNumA, T aNumB,bool &IsEqual){
    if(aNumA == aNumB){
        IsEqual=true;
    }else{
        IsEqual=false;
    }
}

template<typename T,typename X,typename Y>
void
TestComplexDispatch(DataType *aNumA,DataType *aNumB,DataType *aNumC){
    T *data_one=(T*)aNumA->GetData();
    X *data_two=(X*)aNumB->GetData();
    Y *data_out=(Y*)aNumC->GetData();

    REQUIRE(aNumA->GetSize()==aNumB->GetSize());

    for(auto i=0;i<aNumA->GetSize();i++){
        REQUIRE(data_out[i]==data_one[i]+data_two[i]);
    }
}


template<typename T>
void
GenerateData(DataType *aDataType,double aVal){
    T *data=(T*)aDataType->GetData();
    auto size=aDataType->GetSize();
    for(auto i=0;i<size;i++){
        data[i]=aVal;
    }
}

template<typename T>
void
ChangeType(DataType *aDataType,Precision aPrecision){
    auto size=aDataType->GetSize();
    T* temp=new T[size];
    aDataType->SetData((char*)temp);
    aDataType->SetPrecision(aPrecision);
    aDataType->Init<T>();

}

void
TEST_DISPATCHER(){

    bool rc=false;
    int a=5;
    int b=10;
    Precision precision=INT;
    SIMPLE_DISPATCH(precision, TestSimpleDispatch,a,b,rc)
    REQUIRE(rc==false);

    a=10;
    SIMPLE_DISPATCH(precision, TestSimpleDispatch,a,b,rc)
    REQUIRE(rc==true);
    rc=false;

    float temp_float_a=10.332;
    SIMPLE_DISPATCH(precision, TestSimpleDispatch,temp_float_a,b,rc)
    REQUIRE(rc==true);

    float temp_float_b=10.332;
    rc=false;
    precision=FLOAT;
    SIMPLE_DISPATCH(precision, TestSimpleDispatch,temp_float_a,temp_float_b,rc)
    REQUIRE(rc==true);

    DataType dataA(50,FLOAT);
    DataType dataB(50,FLOAT);
    DataType dataOut(50,FLOAT);

    SIMPLE_DISPATCH(FLOAT, GenerateData,&dataOut,3)

    precision=FFF;
    DISPATCHER(precision, TestComplexDispatch,&dataA,&dataB,&dataOut)




}


TEST_CASE("Dispatcher Test", "[Dispatcher]") {
    TEST_DISPATCHER();
}

