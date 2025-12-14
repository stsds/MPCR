/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <operations/BasicOperations.hpp>
#include <libraries/catch/catch.hpp>
#include <utilities/MPCRDispatcher.hpp>


using namespace mpcr::operations;
using namespace mpcr::precision;
using namespace std;


void
TEST_BASIC_OPERATION() {
    SECTION("Testing Sweep") {
        cout << "Testing Basic Utilities ..." << endl;
        cout << "Testing Sweep ..." << endl;

        DataType a(4, 6, FLOAT);
        DataType b(6, FLOAT);
        DataType c(FLOAT);
        int margin = 2;


        auto data_one = (float *) a.GetData();
        auto data_two = (float *) b.GetData();
        auto size = a.GetSize();

        for (auto i = 0; i < size; i++) {
            data_one[ i ] = 0;
        }
        for (auto i = 0; i < b.GetSize(); i++) {
            data_two[ i ] = i + 1;
        }

        auto validator = new float[size];
        int val = 1;
        auto col = a.GetNCol();
        auto row = a.GetNRow();
        for (auto i = 0; i < size; i++) {
            validator[ i ] = val;
            val++;
            if (val > col) {
                val = 1;
            }
        }

        DISPATCHER(FFF, basic::Sweep, a, b, c, margin, "+")

        auto temp_out = (float *) c.GetData();
        auto itr = 0;
        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                auto idx = ( j * row ) + i;
                REQUIRE(validator[ itr ] == temp_out[ idx ]);
                itr++;
            }
        }

        delete[] validator;

        cout << "Testing Sweep Vector ..." << endl;

        DataType sweep_vec(24, "float");
        a.ToVector();
        data_one = (float *) a.GetData();
        data_two = (float *) sweep_vec.GetData();
        for (auto i = 0; i < size; i++) {
            data_one[ i ] = i;
            data_two[ i ] = i;
        }
        margin = 1;
        DISPATCHER(FFF, basic::Sweep, a, sweep_vec, c, margin, "*")
        temp_out = (float *) c.GetData();
        size = c.GetSize();
        REQUIRE(size == 24);
        REQUIRE(c.IsMatrix() == false);

        for (auto i = 0; i < size; i++) {
            REQUIRE(temp_out[ i ] == data_one[ i ] * data_two[ i ]);
        }


    }SECTION("Testing Sweep With Small stat size & Margin one") {

        DataType a(5, 5, FLOAT);
        DataType b(3, DOUBLE);

        DataType result(DOUBLE);

        int margin = 1;

        auto data_one = (float *) a.GetData();
        auto data_two = (double *) b.GetData();
        auto size_in_a = a.GetSize();
        auto size_in_b = b.GetSize();

        for (auto i = 0; i < size_in_a; i++) {
            data_one[ i ] = 1;
        }
        for (auto i = 0; i < size_in_b; i++) {
            data_two[ i ] = i % 3;
        }

        auto validator = new double[size_in_a];
        auto counter = 0;
        for (auto i = 0; i < size_in_a; i++) {
            validator[ i ] = counter;
            counter++;
            if (counter == 3) {
                counter = 0;
            }
        }

        DISPATCHER(FDD, basic::Sweep, a, b, result, margin, "*")

        auto temp_out = (double *) result.GetData();
        for (auto i = 0; i < size_in_a; i++) {
            REQUIRE(validator[ i ] == temp_out[ i ]);
        }

        delete[] validator;
    }SECTION("Testing Sweep with Margin two") {
        DataType a(5, 5, FLOAT);
        DataType b(5, DOUBLE);

        DataType result(DOUBLE);
        int margin = 2;

        auto data_one = (float *) a.GetData();
        auto data_two = (double *) b.GetData();
        auto size_in_a = a.GetSize();
        auto size_in_b = b.GetSize();

        for (auto i = 0; i < size_in_a; i++) {
            data_one[ i ] = 1;
        }
        for (auto i = 0; i < size_in_b; i++) {
            data_two[ i ] = i % 5;
        }

        auto validator = new double[size_in_a];
        for (auto i = 0; i < size_in_a; i++) {
            validator[ i ] = i % 5;
        }

        DISPATCHER(FDD, basic::Sweep, a, b, result, margin, "*")

        auto temp_out = (double *) result.GetData();
        auto col = a.GetNCol();
        auto row = a.GetNRow();
        auto counter = 0;
        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                auto idx = ( j * row ) + i;
                REQUIRE(validator[ counter ] == temp_out[ idx ]);
                counter++;
            }
        }

        delete[] validator;

    }SECTION("Testing Min/Max") {
        cout << "Testing Min/Max With Index ..." << endl;

        DataType a(50, FLOAT);
        DataType output(FLOAT);
        size_t index = 0;

        auto data_in_a = (float *) a.GetData();
        data_in_a[ 20 ] = -15;

        SIMPLE_DISPATCH(FLOAT, basic::MinMax, a, output, index, false)
        auto data_out = (float *) output.GetData();

        REQUIRE(output.GetSize() == 1);
        REQUIRE(index == 20);
        REQUIRE(data_out[ 0 ] == -15);

        data_in_a[ 15 ] = 200;
        index = 0;

        SIMPLE_DISPATCH(FLOAT, basic::MinMax, a, output, index, true)
        data_out = (float *) output.GetData();

        REQUIRE(output.GetSize() == 1);
        REQUIRE(index == 15);
        REQUIRE(data_out[ 0 ] == 200);
    }SECTION("Test Get Diagonal") {
        cout << "Testing Get Diagonal ..." << endl;
        DataType a(5, 5, FLOAT);
        DataType output(FLOAT);

        auto data_in_a = (float *) a.GetData();
        REQUIRE(a.GetDimensions() != nullptr);
        auto num_row = a.GetDimensions()->GetNRow();
        auto num_col = a.GetDimensions()->GetNCol();

        for (auto i = 0; i < num_row; ++i) {
            for (auto j = 0; j < num_col; j++) {
                data_in_a[ ( i * num_col ) + j ] = i;
            }
        }

        SIMPLE_DISPATCH(FLOAT, basic::GetDiagonal, a, output)

        auto data_out = (float *) output.GetData();
        auto size_out = output.GetSize();

        REQUIRE(size_out == 5);
        for (auto i = 0; i < size_out; ++i) {
            REQUIRE(data_out[ i ] == i);
        }

        a.ToVector();
        Dimensions a_dims(5, 5);
        SIMPLE_DISPATCH(FLOAT, basic::GetDiagonal, a, output, &a_dims)

        data_out = (float *) output.GetData();
        size_out = output.GetSize();

        REQUIRE(size_out == 5);
        for (auto i = 0; i < size_out; ++i) {
            REQUIRE(data_out[ i ] == i);
        }


    }SECTION("Test Checking Types") {
        cout << "Testing Type Checks ..." << endl;
        DataType a(FLOAT);
        DataType b(DOUBLE);
#ifdef USE_CUDA
        DataType c(HALF,GPU);
#else
        DataType c(HALF);
#endif
        REQUIRE(basic::IsSFloat(a) == false);
        REQUIRE(basic::IsFloat(a) == true);
        REQUIRE(basic::IsDouble(a) == false);


        REQUIRE(basic::IsSFloat(b) == false);
        REQUIRE(basic::IsFloat(b) == false);
        REQUIRE(basic::IsDouble(b) == true);

#ifdef USE_CUDA
        REQUIRE(basic::IsSFloat(c) == true);
        REQUIRE(basic::IsFloat(c) == false);
        REQUIRE(basic::IsDouble(c) == false);
#else
        REQUIRE(basic::IsSFloat(c) == false);
        REQUIRE(basic::IsFloat(c) == true);
        REQUIRE(basic::IsDouble(c) == false);
#endif


    }SECTION("Testing CBind Same Precision") {
        cout << "Testing CBind ..." << endl;
        DataType a(6, 4, FLOAT);
        DataType b(6, 4, FLOAT);
        DataType c(FLOAT);

        auto data_in_a = (float *) a.GetData();
        auto data_in_b = (float *) b.GetData();
        auto size_a = a.GetSize();

        size_t counter = 0;
        for (auto i = 0; i < size_a; i++) {
            data_in_a[ i ] = counter;
            data_in_b[ i ] = counter;
            counter++;
        }


        DISPATCHER(FFF, basic::ColumnBind, a, b, c)

        DataType test(6, 8, FLOAT);

        auto size = c.GetSize();
        auto temp_data = (float *) test.GetData();
        auto temp_data_in_c = (float *) c.GetData();
        REQUIRE(size == ( 6 * 8 ));

        counter = 0;

        for (auto i = 0; i < size; ++i) {
            REQUIRE(counter == temp_data_in_c[ i ]);
            counter++;
            if (counter == 24) {
                counter = 0;
            }
        }

    }SECTION("Testing CBind Different Precision") {
        DataType a(6, 4, FLOAT);
        DataType b(6, 4, DOUBLE);
        DataType c(DOUBLE);

        auto data_in_a = (float *) a.GetData();
        auto data_in_b = (double *) b.GetData();
        auto size_a = a.GetSize();

        size_t counter = 0;
        for (auto i = 0; i < size_a; i++) {
            data_in_a[ i ] = counter;
            data_in_b[ i ] = counter;
            counter++;
        }

        DISPATCHER(FDD, basic::ColumnBind, a, b, c)

        DataType test(6, 8, DOUBLE);
        auto temp_data = (double *) test.GetData();
        auto temp_data_in_c = (double *) c.GetData();
        auto size = c.GetSize();
        REQUIRE(size == ( 6 * 8 ));

        counter = 0;
        for (auto i = 0; i < size; ++i) {
            REQUIRE(counter == temp_data_in_c[ i ]);
            counter++;
            if (counter == 24) {
                counter = 0;
            }
        }

    }SECTION("Testing RBind Same Precision") {
        cout << "Testing RBind ..." << endl;

        DataType a(6, 4, FLOAT);
        DataType b(6, 4, FLOAT);
        DataType c(FLOAT);

        auto data_in_a = (float *) a.GetData();
        auto data_in_b = (float *) b.GetData();

        auto counter = 0;
        auto row = a.GetNRow();
        auto col = a.GetNCol();

        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                auto idx = ( j * row ) + i;
                data_in_a[ idx ] = counter;
                counter++;
            }
        }

        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                auto idx = ( j * row ) + i;
                data_in_b[ idx ] = counter;
                counter++;
            }
        }


        DISPATCHER(FFF, basic::RowBind, a, b, c)

        DataType test(12, 4, FLOAT);
        auto temp_data = (float *) test.GetData();
        auto temp_data_in = (float *) c.GetData();
        auto size = c.GetSize();
        REQUIRE(size == ( 12 * 4 ));
        row = c.GetNRow();
        col = c.GetNCol();

        counter = 0;
        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                auto idx = ( j * row ) + i;
                REQUIRE(temp_data_in[ idx ] == counter);
                counter++;
            }
        }

    }SECTION("Testing RBind Different Precision") {
        DataType a(6, 4, FLOAT);
        DataType b(6, 4, DOUBLE);
        DataType c(DOUBLE);

        for (auto i = 0; i < 6*4; ++i) {
            REQUIRE(a.GetVal( i ) ==0);
            REQUIRE(b.GetVal( i ) ==0);
        }

        DISPATCHER(FDD, basic::RowBind, a, b, c)

        DataType test(12, 4, DOUBLE);
        auto temp_data = (double *) test.GetData();
        auto temp_data_in = (double *) c.GetData();
        auto size = c.GetSize();

        REQUIRE(size == ( 12 * 4 ));
        for (auto i = 0; i < size; ++i) {
            REQUIRE(temp_data[ i ] == temp_data_in[ i ]);
        }

    }SECTION("Testing Replicate") {
        cout << "Testing Replicate ..." << endl;
        DataType a(5, FLOAT);
        auto data_in_a = (float *) a.GetData();
        auto size_in_a = a.GetSize();

        for (auto i = 0; i < size_in_a; ++i) {
            data_in_a[ i ] = i;
        }

        DataType b(FLOAT);
        SIMPLE_DISPATCH(FLOAT, basic::Replicate, a, b, 50)
        auto size = b.GetSize();
        REQUIRE(size == 50);
        auto data_out = (float *) b.GetData();

        for (auto i = 0; i < size; ++i) {
            REQUIRE(data_out[ i ] == i % 5);
        }
    }SECTION("Test NA Omit") {
        cout << "Testing NA Replace ..." << endl;
        DataType a(50, FLOAT);

        auto data_in_a = (float *) a.GetData();
        auto size = a.GetSize();
        float zero = 0;
        for (auto i = 30; i < size - 10; i++) {
            data_in_a[ i ] = 0;
            data_in_a[ i ] = data_in_a[ i ] / zero;
        }

        SIMPLE_DISPATCH(FLOAT, basic::NAReplace, a, 3.5)

        for (auto i = 30; i < size - 10; i++) {
            REQUIRE(data_in_a[ i ] == 3.5);
        }

        for (auto i = 30; i < size - 10; i++) {
            data_in_a[ i ] = 0;
            data_in_a[ i ] = data_in_a[ i ] / zero;
        }

        cout << "Testing NA Omit ..." << endl;
        SIMPLE_DISPATCH(FLOAT, basic::NAExclude, a)

        data_in_a = (float *) a.GetData();
        size = a.GetSize();
        REQUIRE(size == 40);
        for (auto i = 0; i < size; i++) {
            REQUIRE(data_in_a[ i ] == 0);
        }

        cout << "Testing NA Omit in Matrix ..." << endl;
        DataType b(5, 6, FLOAT);

        auto data_in_b = (float *) b.GetData();
        auto size_in_b = b.GetSize();
        auto col = b.GetNCol();
        auto row = b.GetNRow();
        auto count = 0;

        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                auto idx = ( j * row ) + i;
                data_in_b[ idx ] = count;
                count++;
            }
        }

        b.SetValMatrix(2, 2, NAN);
        b.SetValMatrix(3, 2, NAN);

        SIMPLE_DISPATCH(FLOAT, basic::NAExclude, b)

        data_in_b = (float *) b.GetData();

        REQUIRE(b.GetNCol() == 6);
        REQUIRE(b.GetNRow() == 3);
        size = b.GetSize();
        REQUIRE(size == ( 5 * 6 ) - ( 12 ));
        bool flag;


        for (auto i = 0; i < size; i++) {
            flag = ( data_in_b[ i ] >= 24 || data_in_b[ i ] <= 11 );
            REQUIRE(flag == true);
        }


    }SECTION("Test Object Size") {
        cout << "Testing Get Object Size ..." << endl;
        DataType a(50, FLOAT);
        auto size = sizeof(bool) + sizeof(Precision) + sizeof(size_t);
        size += ( 50 * sizeof(float));
        size+= sizeof (DataHolder);

        REQUIRE(size == a.GetObjectSize());

        a.ToMatrix(5, 10);
        size += ( 2 * sizeof(size_t));

        REQUIRE(size == a.GetObjectSize());
    }SECTION("Test Concatenate") {
        cout << "Testing Concatenate ..." << endl;
        vector <DataType *> mpr_objects;

        /** Size Must be Even Number. Odd Number is Handled in R Adapter**/
        size_t size = 12;
        REQUIRE(size % 2 == 0);
        mpr_objects.resize(size);

        vector <Precision> precisions{ FLOAT, DOUBLE};

        for (auto i = 0; i < size; i++) {
            auto temp_mpr = new DataType(30, precisions[ i % 2 ]);
            mpr_objects[ i ] = temp_mpr;
        }

        size_t size_out = 0;
        auto precision_out = HALF;

        for (auto i = 0; i < size; i++) {
            size_out += mpr_objects[ i ]->GetSize();
            precision_out = GetOutputPrecision(precision_out,
                                               mpr_objects[ i ]->GetPrecision());

        }

        auto pOutput = new DataType(size_out, precision_out);
        auto data_out = (double *) pOutput->GetData();

        for (auto i = 0; i < size_out; i++) {
            data_out[ i ] = 0;
        }
        auto operation_precision = HALF;
        auto precision_one = HALF;
        auto precision_two = HALF;
        size_t offset = 0;


        for (auto i = 0; i < size; i += 2) {
            precision_one = mpr_objects[ i ]->GetPrecision();
            precision_two = mpr_objects[ i + 1 ]->GetPrecision();
            operation_precision = GetOperationPrecision(precision_one,
                                                        precision_two,
                                                        precision_out);

            DISPATCHER(operation_precision, basic::Concatenate,
                       *mpr_objects[ i ],
                       *mpr_objects[ i + 1 ], *pOutput, offset)

        }

        REQUIRE(pOutput->GetSize() == size_out);
        REQUIRE(pOutput->GetPrecision() == DOUBLE);
        REQUIRE(offset == pOutput->GetSize());

        delete pOutput;
        for (auto &x: mpr_objects) {
            delete x;
        }
    }SECTION("Test Scale and Center") {
        DataType a(5, 6, FLOAT);
        DataType scale_center(6, DOUBLE);


        auto data_in_a = (float *) a.GetData();
        auto size_in_a = a.GetSize();

        auto data_in_scale_center = (double *) scale_center.GetData();
        auto size_in_scale_center = scale_center.GetSize();

        for (auto i = 0; i < size_in_a; i++) {
            data_in_a[ i ] = i;
        }


        for (auto i = 0; i < size_in_scale_center; i++) {
            data_in_scale_center[ i ] = i + 1;
        }

        DataType output(DOUBLE);
        DISPATCHER(FDD, basic::ApplyCenter, a, scale_center, output)

        REQUIRE(output.GetSize() == 30);
        REQUIRE(output.GetNCol() == 6);
        REQUIRE(output.GetNRow() == 5);
        auto output_size = output.GetSize();
        auto data_in_output = (double *) output.GetData();

        for (auto i = 0; i < output_size; i++) {
            REQUIRE(data_in_output[ i ] == data_in_a[ i ] -
                                           data_in_scale_center[ i %
                                                                 size_in_scale_center ]);
        }

        auto validator = new double[output_size];
        memcpy((char *) validator, (char *) data_in_output,
               sizeof(double) * output_size);

        DISPATCHER(DDD, basic::ApplyScale, output, scale_center, output)
        output_size = output.GetSize();
        REQUIRE(output_size == 30);
        REQUIRE(output.GetNCol() == 6);
        REQUIRE(output.GetNRow() == 5);
        data_in_output = (double *) output.GetData();

        for (auto i = 0; i < output_size; i++) {
            REQUIRE(data_in_output[ i ] == validator[ i ] /
                                           data_in_scale_center[ i %
                                                                 size_in_scale_center ]);
        }

        delete[] validator;
        output.ClearUp();

        auto calc_mean = true;
        auto row = a.GetNRow();
        auto col = a.GetNCol();
        validator = new double[row];

        double accum;
        size_t start_idx;
        double zero = 0;
        a.SetVal(3, zero);
        a.SetVal(3, zero / zero);
        auto counter = 0;
        for (auto i = 0; i < row; i++) {
            accum = 0;
            counter = 0;

            for (auto j = 0; j < col; j++) {
                start_idx = ( j * row ) + i;
                if (!isnan(data_in_a[ start_idx ])) {
                    accum += data_in_a[ start_idx ];
                    counter++;
                }
            }

            validator[ i ] = accum / counter;
        }

        DISPATCHER(FDD, basic::ApplyCenter, a, scale_center, output, &calc_mean)

        data_in_output = (double *) output.GetData();
        REQUIRE(output_size == 30);
        REQUIRE(output.GetNCol() == 6);
        REQUIRE(output.GetNRow() == 5);

        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                start_idx = ( j * row ) + i;
                if (!isnan(data_in_a[ start_idx ])) {
                    REQUIRE(data_in_output[ start_idx ] ==
                            data_in_a[ start_idx ] - validator[ i ]);
                }
            }
        }

        for (auto i = 0; i < output_size; i++) {
            data_in_output[ i ] = 0;
            data_in_a[ i ] = i;
        }
        calc_mean = false;
        DISPATCHER(FDD, basic::ApplyCenter, a, scale_center, output, &calc_mean)
        data_in_output = (double *) output.GetData();
        REQUIRE(output_size == 30);
        REQUIRE(output.GetNCol() == 6);
        REQUIRE(output.GetNRow() == 5);

        for (auto i = 0; i < output_size; i++) {
            REQUIRE(data_in_output[ i ] == data_in_a[ i ]);
        }

        delete[] validator;


        auto standard_deviation = 1.870828693387;
        auto perc = 0.001;
        row = output.GetNRow();
        col = output.GetNCol();
        counter = 0;
        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                auto idx = ( j * row ) + i;
                data_in_output[ idx ] = counter;
                data_in_a[ idx ] = counter;
                counter++;
            }
        }

//        for (auto i = 0; i < output_size; i++) {
//            data_in_output[ i ] = i;
//            data_in_a[ i ] = i;
//        }

        auto calc_sd = true;
        DISPATCHER(FDD, basic::ApplyScale, a, scale_center, output, &calc_sd)

        data_in_output = (double *) output.GetData();
        REQUIRE(output_size == 30);
        REQUIRE(output.GetNCol() == 6);
        REQUIRE(output.GetNRow() == 5);

        for (auto i = 0; i < row; i++) {
            for (auto j = 0; j < col; j++) {
                start_idx = ( j * row ) + i;
                auto val_1 = data_in_a[ start_idx ] / standard_deviation;
                auto val_2 = data_in_output[ start_idx ];
                if (val_1 != 0) {
                    auto val_3 = abs(( val_2 - val_1 )) / val_1;
                    REQUIRE(val_3 < perc);
                } else {
                    REQUIRE(val_2 == 0);
                }
            }
        }

        for (auto i = 0; i < output_size; i++) {
            data_in_output[ i ] = i;
            data_in_a[ i ] = i;
        }

        calc_sd = false;
        DISPATCHER(FDD, basic::ApplyScale, a, scale_center, output, &calc_sd)
        data_in_output = (double *) output.GetData();
        REQUIRE(output_size == 30);
        REQUIRE(output.GetNCol() == 6);
        REQUIRE(output.GetNRow() == 5);
        for (auto i = 0; i < output_size; i++) {
            REQUIRE(data_in_output[ i ] == data_in_a[ i ]);
        }

    }

}


TEST_CASE("BasicOperations", "[BasicOperations]") {
    TEST_BASIC_OPERATION();
}
