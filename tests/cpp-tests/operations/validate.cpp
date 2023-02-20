
#include <libraries/catch/catch.hpp>
#include <utilities/MPRDispatcher.hpp>
#include <operations/LinearAlgebra.hpp>


using namespace std;
using namespace mpr::precision;
using namespace mpr::operations;


void
svd() {

    cout << "---------- SVD ---------" << endl;
    vector <double> values = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 1, 1, 1};
    DataType a(values, FLOAT);
    a.ToMatrix(9, 4);




    DataType d(FLOAT);
    DataType u(FLOAT);
    DataType v(FLOAT);

    SIMPLE_DISPATCH(FLOAT, linear::SVD, a, d, u, v, a.GetNCol(),a.GetNCol())

    cout << "---------- Input ---------" << endl;
    a.Print();

    cout << "---------- D ---------" << endl;
    d.Print();

    cout << "---------- U ---------" << endl;
    u.Print();

    cout << "---------- V ---------" << endl;
    v.Print();


}

void
QRQ(){
    cout << "---------- QR.Q ---------" << endl;

    vector <double> values = {1, 2, 3, 2, 4, 6, 3, 3, 3};
    DataType a(values, FLOAT);
    a.ToMatrix(3, 3);
    cout << "---------- Input ---------" << endl;
    a.Print();

    DataType qraux(FLOAT);
    DataType pivot(FLOAT);
    DataType qr(FLOAT);
    size_t rank = 0;

    SIMPLE_DISPATCH(FLOAT, linear::QRDecomposition, a, qr, qraux, pivot,
                    rank)


    DataType qr_q(FLOAT);
    cout << "---------- QR ---------" << endl;
    qr.Print();
    cout << "---------- Qraux---------" << endl;
    qraux.Print();

    SIMPLE_DISPATCH(FLOAT, linear::QRDecompositionQ, qr, qraux, qr_q, FALSE)
    cout << "--------- We need to validate R -------" << endl;
    cout << "---------- R---------" << endl;
    qr_q.Print();
}

void
RCOND(){
    cout << " R Cond ..." << endl;
    vector <double> values = {100, 2, 3, 3, 2, 1, 300, 3, 3, 400, 5, 6, 4,
                              44, 56, 1223};
    DataType a(values, FLOAT);
    a.ToMatrix(4, 4);
    cout << "---------- Input ---------" << endl;
    a.Print();


    DataType b(FLOAT);
    SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "I", true)

    cout << "---------- rcond 'I' Upper Triangle =true  ---------" << endl;
    b.Print();


    b.ClearUp();

    SIMPLE_DISPATCH(FLOAT, linear::ReciprocalCondition, a, b, "O", true)
    cout << "---------- rcond 'O' Upper Triangle =true  ---------" << endl;
    b.Print();

}


TEST_CASE("LinearAlgebra", "[Linear Algebra]") {
//svd();
//QRQ();
RCOND();
}
