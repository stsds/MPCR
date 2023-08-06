

#include <adapters/RLinearAlgebra.hpp>
#include <adapters/RHelpers.hpp>
#include <data-units/Promoter.hpp>
#include <utilities/MPRDispatcher.hpp>


using namespace mpr::operations;


DataType *
RTrsm(DataType *aInputA, DataType *aInputB, const bool &aUpperTri,
      const bool &aTranspose, const char &aSide, const double &aAlpha) {

    Promoter pr(2);
    pr.Insert(*aInputA);
    pr.Insert(*aInputB);
    pr.Promote();

    auto precision = aInputA->GetPrecision();
    auto pOutput = new DataType(precision);

    SIMPLE_DISPATCH(precision, linear::BackSolve, *aInputA, *aInputB, *pOutput,
                    aInputA->GetNCol(), aUpperTri, aTranspose, aSide, aAlpha)

    pr.DePromote();

    return pOutput;
}


void
RGemm(DataType *aInputA, SEXP aInputB, DataType *aInputC,
      const bool &aTransposeA, const bool &aTransposeB, const double &aAlpha,
      const double &aBeta) {

    bool aSingle = ((SEXP) aInputB == R_NilValue );
    Promoter pr(3);
    DataType *temp_b = nullptr;

    if (aSingle) {
        DataType dump(0, aInputA->GetPrecision());
        temp_b = &dump;
    } else {
        temp_b = (DataType *) Rcpp::internal::as_module_object_internal(
            aInputB);
        if (!temp_b->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MMPR Object",
                -1);
        }

    }
    pr.Insert(*aInputA);
    pr.Insert(*temp_b);
    pr.Insert(*aInputC);
    pr.Promote();

    auto precision = aInputA->GetPrecision();
    SIMPLE_DISPATCH(precision, linear::CrossProduct, *aInputA, *temp_b,
                    *aInputC, aTransposeA, aTransposeB, true, aAlpha, aBeta)

    pr.DePromote();
}


DataType *
RCrossProduct(DataType *aInputA, SEXP aInputB) {

    // cross(x,y) -> x y
    // cross(x) -> t(x) x

    bool aSingle = ((SEXP) aInputB == R_NilValue );
    Promoter pr(2);
    auto transpose = false;
    DataType *temp_b = nullptr;

    if (aSingle) {
        DataType dump(0, aInputA->GetPrecision());
        temp_b = &dump;
        transpose = true;
    } else {
        temp_b = (DataType *) Rcpp::internal::as_module_object_internal(
            aInputB);
        if (!temp_b->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MMPR Object",
                -1);
        }
        pr.Insert(*aInputA);
        pr.Insert(*temp_b);
        pr.Promote();
    }

    auto precision = aInputA->GetPrecision();

    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, linear::CrossProduct, *aInputA, *temp_b,
                    *pOutput,
                    transpose, false)

    if (!aSingle) {
        pr.DePromote();
    }

    return pOutput;
}


DataType *
RTCrossProduct(DataType *aInputA, SEXP aInputB) {


    // tcross(x,y) -> x t(y)
    // tcross(x) -> x t(x)

    bool aSingle = ((SEXP) aInputB == R_NilValue );
    Promoter pr(2);
    DataType *temp_b = nullptr;

    if (aSingle) {
        DataType dump(0, aInputA->GetPrecision());
        temp_b = &dump;
    } else {
        temp_b = (DataType *) Rcpp::internal::as_module_object_internal(
            aInputB);
        if (!temp_b->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MMPR Object",
                -1);
        }
        pr.Insert(*aInputA);
        pr.Insert(*temp_b);
        pr.Promote();
    }

    auto precision = aInputA->GetPrecision();

    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, linear::CrossProduct, *aInputA, *temp_b,
                    *pOutput,
                    false, true)

    if (!aSingle) {
        pr.DePromote();
    }

    return pOutput;

}


bool
RIsSymmetric(DataType *aInputA) {

    bool output = false;
    SIMPLE_DISPATCH(aInputA->GetPrecision(), linear::IsSymmetric, *aInputA,
                    output)
    return output;

}


DataType *
RBackSolve(DataType *aInputA, DataType *aInputB, const long &aCol,
           const bool &aUpperTriangle, const bool &aTranspose) {
    auto col = aCol;
    if (aCol < 0) {
        col = aInputA->GetNCol();
    }
    Promoter pr(2);
    pr.Insert(*aInputA);
    pr.Insert(*aInputB);
    pr.Promote();

    auto precision = aInputA->GetPrecision();
    auto pOutput = new DataType(precision);

    SIMPLE_DISPATCH(precision, linear::BackSolve, *aInputA, *aInputB, *pOutput,
                    col, aUpperTriangle, aTranspose)

    pr.DePromote();

    return pOutput;
}


DataType *
RCholesky(DataType *aInputA, const bool &aUpperTriangle) {
    auto precision = aInputA->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, linear::Cholesky, *aInputA, *pOutput,
                    aUpperTriangle)
    return pOutput;

}


DataType *
RCholeskyInv(DataType *aInputA, const size_t &aSize) {
    auto precision = aInputA->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, linear::CholeskyInv, *aInputA, *pOutput, aSize)
    return pOutput;
}


DataType *
RSolve(DataType *aInputA, SEXP aInputB) {

    bool aSingle = ((SEXP) aInputB == R_NilValue );
    Promoter pr(2);
    DataType *temp_b = nullptr;

    if (aSingle) {
        DataType dump(0, aInputA->GetPrecision());
        temp_b = &dump;
    } else {
        temp_b = (DataType *) Rcpp::internal::as_module_object_internal(
            aInputB);
        if (!temp_b->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MMPR Object",
                -1);
        }
        pr.Insert(*aInputA);
        pr.Insert(*temp_b);
        pr.Promote();
    }

    auto precision = aInputA->GetPrecision();

    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, linear::Solve, *aInputA, *temp_b,
                    *pOutput, aSingle)

    if (!aSingle) {
        pr.DePromote();
    }

    return pOutput;

}


std::vector <DataType>
RSVD(DataType *aInputA, const long &aNu, const long &aNv,
     const bool &aTranspose) {

    auto row = aInputA->GetNRow();
    auto col = aInputA->GetNCol();
    auto nv = aNv;
    auto nu = aNu;

    if (aNv < 0) {
        nv = std::min(row, col);
    }
    if (aNu < 0) {
        nu = std::min(row, col);
    }


    auto precision = aInputA->GetPrecision();

    auto d = new DataType(precision);
    auto u = new DataType(precision);
    auto v = new DataType(precision);


    SIMPLE_DISPATCH(precision, linear::SVD, *aInputA, *d, *u, *v, nu, nv,
                    aTranspose)

    std::vector <DataType> output;

    output.push_back(*d);
    output.push_back(*u);
    output.push_back(*v);

    return output;

}


DataType *
RTranspose(DataType *aInputA) {
    auto pOutput = new DataType(*aInputA);
    pOutput->Transpose();
    return pOutput;
}


DataType *
RNorm(DataType *aInputA, const std::string &aType) {
    auto precision = aInputA->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, linear::Norm, *aInputA, aType, *pOutput)
    return pOutput;
}


std::vector <DataType>
RQRDecomposition(DataType *aInputA, const double &aTolerance) {

    auto precision = aInputA->GetPrecision();

    auto qr = new DataType(precision);
    auto qraux = new DataType(precision);
    auto pivot = new DataType(precision);
    auto rank = new DataType(precision);

    SIMPLE_DISPATCH(precision, linear::QRDecomposition, *aInputA, *qr, *qraux,
                    *pivot, *rank, aTolerance)

    std::vector <DataType> output;
    output.push_back(*qr);
    output.push_back(*qraux);
    output.push_back(*pivot);
    output.push_back(*rank);

    return output;

}


DataType *
RRCond(DataType *aInputA, const std::string &aNorm, const bool &aTriangle) {

    auto row = aInputA->GetNRow();
    auto col = aInputA->GetNCol();
    auto precision = aInputA->GetPrecision();
    DataType *temp_input = nullptr;
    auto flag_creation = false;


    if (row != col) {
        DataType temp = *aInputA;
        DataType out(precision);
        if (row < col) {
            temp.Transpose();
        }
        auto qr = new DataType(precision);
        auto qraux = new DataType(precision);
        auto pivot = new DataType(precision);
        auto rank = new DataType(precision);

        SIMPLE_DISPATCH(precision, linear::QRDecomposition, *aInputA, *qr,
                        *qraux,
                        *pivot, *rank)

        temp_input = RQRDecompositionR(qr, false);

        delete qr;
        delete qraux;
        delete pivot;
        delete rank;

        flag_creation = true;

    } else {
        temp_input = aInputA;
    }

    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, linear::ReciprocalCondition, *temp_input,
                    *pOutput,
                    aNorm, aTriangle)

    if (flag_creation) {
        delete temp_input;
    }

    return pOutput;
}


DataType *
RQRDecompositionR(DataType *aInputA, const bool &aComplete) {
    auto precision = aInputA->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, linear::QRDecompositionR, *aInputA, *pOutput,
                    aComplete)
    return pOutput;
}


DataType *
RQRDecompositionQ(DataType *aInputA, DataType *aInputB, const bool &aComplete,
                  SEXP aDvec) {

    auto precision = aInputA->GetPrecision();
    auto pOutput = new DataType(precision);
    if (aDvec == R_NilValue) {
        SIMPLE_DISPATCH(precision, linear::QRDecompositionQ, *aInputA, *aInputB,
                        *pOutput, aComplete)
    } else {
        auto temp_dvec = (DataType *) Rcpp::internal::as_module_object_internal(
            aDvec);
        if (!temp_dvec->IsDataType()) {
            MPR_API_EXCEPTION(
                "Undefined Object . Make Sure You're Using MMPR Object",
                -1);
        }
        SIMPLE_DISPATCH(precision, linear::QRDecompositionQY, *aInputA,
                        *aInputB, *temp_dvec, *pOutput, aComplete)
    }

    return pOutput;

}


DataType *
RQRDecompositionQy(DataType *aInputA, DataType *aInputB, DataType *aDvec) {
    auto precision = aInputA->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, linear::QRDecompositionQY, *aInputA, *aInputB,
                    *aDvec, *pOutput, false)


    return pOutput;
}


DataType *
RQRDecompositionQty(DataType *aInputA, DataType *aInputB, DataType *aDvec) {
    auto precision = aInputA->GetPrecision();
    auto pOutput = new DataType(precision);
    SIMPLE_DISPATCH(precision, linear::QRDecompositionQY, *aInputA, *aInputB,
                    *aDvec, *pOutput, true)


    return pOutput;
}


std::vector <DataType>
REigen(DataType *aInputA, const bool &aOnlyValues) {

    /**
     * This if condition is added since MKL eigen routine has a bug with float
     * affecting the runtime of the function.
     *
     * so for this function, any non double matrix will be converted to double to
     * avoid this timing problem.
     *
     **/
    if (aInputA->GetPrecision() != DOUBLE) {
        aInputA->ConvertPrecision(DOUBLE);
    }

    auto precision = aInputA->GetPrecision();
    DataType *pVector = nullptr;
    auto pValues = new DataType(precision);
    std::vector <DataType> output;

    if (!aOnlyValues) {
        pVector = new DataType(precision);
    }

    SIMPLE_DISPATCH(precision, linear::Eigen, *aInputA, *pValues, pVector)

    output.push_back(*pValues);
    if (!aOnlyValues) {
        output.push_back(*pVector);
    }

    return output;


}