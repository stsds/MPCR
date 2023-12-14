/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_DEFINITIONS_HPP
#define MPCR_DEFINITIONS_HPP

namespace mpcr{
    namespace definitions{

        enum OperationPlacement : int {
            GPU = 0,
            CPU = 1
        };

        /**
         * Int Enum Describing the Precision and Operations order
         * of the MPCR (Data Type)object
         **/
        enum Precision : int {
            /** 16-Bit Precision (Will be replaced with sfloat later on) **/
            HALF = 1,
            /** 32-Bit Precision **/
            FLOAT = 2,
            /** 64-Bit Precision **/
            DOUBLE = 3,
            /** Error Code **/
            ERROR = -1,

            /**
             * Operations order for Dispatching Assuming Output Must be the
             *  Same as One of Inputs.
             *  each precision is multiplied by a prime number according to its
             *  position ( Generating a unique value for each operation )
             *  Numbers used (3,5,7)
             **/

            /** in:sfloat ,in:sfloat ,out:sfloat **/
            SSS = 15,
            /** in:float ,in:sfloat ,out:float **/
            FSF = 25,
            /** in:sfloat ,in:float ,out:float **/
            SFF = 27,
            /** in:float ,in:float ,out:float **/
            FFF = 30,
            /** in:double ,in:sfloat ,out:double **/
            DSD = 35,
            /** in:sfloat ,in:double ,out:double **/
            SDD = 39,
            /** in:double ,in:float ,out:double **/
            DFD = 40,
            /** in:float ,in:double ,out:double **/
            FDD = 42,
            /** in:double ,in:double ,out:double **/
            DDD = 45,

            /** This Operation Combinations are used for Concatenation Only **/

            /** in:sfloat ,in:sfloat ,out:double **/
            SSD = 29,
            /** in:sfloat ,in:sfloat ,out:float **/
            SSF = 22,
            /** in:float ,in:float ,out:double **/
            FFD = 37,
            /** in:sfloat ,in:float ,out:double **/
            SFD = 34,
            /** in:float ,in:sfloat ,out:double **/
            FSD = 32
        };
    }
}

#endif //MPCR_DEFINITIONS_HPP
