

#include <blas.hh>
#include <utilities/MPRDispatcher.hpp>
#include <operations/LinearAlgebra.hpp>


using namespace mpr::operations;


template <typename T>
void
linear::CrossProduct(DataType &aInputA, DataType &aInputB,
                     DataType &aOutput,const bool &aTranspose) {

    auto pData_a = (T *) aInputA.GetData();
    auto pData_b = (T *) aInputB.GetData();

    auto row_a = aInputA.GetNRow();
    auto col_a = aInputA.GetNCol();
    size_t row_b ;
    size_t col_b ;

    if(aTranspose){
        row_b=col_a;
        col_b=row_a;
    }else{
        row_b=aInputB.GetNRow();
        col_b=aInputB.GetNCol();
    }

    if (col_a != row_b) {
        MPR_API_EXCEPTION("Wrong Matrix Dimensions", -1);
    }

    auto output_size = row_a * col_b;
    aOutput.ClearUp();
    aOutput.SetSize(output_size);
    aOutput.SetDimensions(row_a, col_b);

    auto pData_out = new T[output_size];

    if(aTranspose){
        blas::gemm(LAYOUT, blas::Op::NoTrans, blas::Op::NoTrans,
                   row_a, col_b, col_a, 1, pData_a, row_a, pData_b, row_b, 0,
                   pData_out, row_a);
    }else{
//        blas::syrk(LAYOUT)
    }

    aOutput.SetData((char *) pData_out);

}


template <typename T>
void linear::IsSymmetric(DataType &aInput, bool &aOutput) {

    aOutput = false;
    auto pData = (T *) aInput.GetData();
    auto col = aInput.GetNCol();
    auto row = aInput.GetNRow();

    if (col != row) {
        return;
    }

    size_t idx_col_maj;
    size_t idx_row_maj;
    auto epsilon = std::numeric_limits <T>::epsilon();
    T val;
    for (auto i = 0; i < col; i++) {
        for (auto j = 0; j < row; j++) {
            if (i == j) {
                break;
            }
            idx_col_maj = ( i * row ) + j;
            idx_row_maj = ( j * col ) + i;
            val = std::fabs(pData[ idx_row_maj ] - pData[ idx_col_maj ]);
            if (val > epsilon) {
                return;
            }
        }
    }

    aOutput = true;

}


SIMPLE_INSTANTIATE(void, linear::CrossProduct, DataType &aInputA,
                   DataType &aInputB, DataType &aOutput,const bool &aTranspose)

SIMPLE_INSTANTIATE(void, linear::IsSymmetric, DataType &aInput, bool &aOutput)

