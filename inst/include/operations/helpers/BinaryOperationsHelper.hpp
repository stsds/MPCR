/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_BINARYOPERATIONSHELPER_HPP
#define MPCR_BINARYOPERATIONSHELPER_HPP

#include <limits.h>


/************************** Operations *******************************/



#define BINARY_OPERATION(dataA, dataB, dataOut, FUN, sizeA, sizeB,sizeOut)     \
           for(auto i=0;i<sizeOut;i++){                                        \
                dataOut[i]=dataA[i%sizeA] FUN dataB[i % sizeB];                \
            }                                                                  \


#define RUN_BINARY_OP(dataA, dataB, dataOut, FUN, sizeA,sizeB,sizeOut)         \
         if(FUN=="+")  {                                                       \
              BINARY_OPERATION(dataA,dataB,dataOut,+,sizeA,sizeB,sizeOut)      \
         }else if(FUN=="-")  {                                                 \
           BINARY_OPERATION(dataA,dataB,dataOut,-,sizeA,sizeB,sizeOut)         \
         }else if(FUN=="*")  {                                                 \
           BINARY_OPERATION(dataA,dataB,dataOut,*,sizeA,sizeB,sizeOut)         \
         }else if(FUN=="/")  {                                                 \
           BINARY_OPERATION(dataA,dataB,dataOut,/,sizeA,sizeB,sizeOut)         \
         }else if(FUN=="^")  {                                                 \
           for(auto i=0;i<sizeOut;i++){                                        \
                dataOut[i]=std::pow(dataA[i%sizeA],dataB[i%sizeB]);            \
            }                                                                  \
         }else {                                                               \
             MPCR_API_EXCEPTION("Operation Not Supported", -1);                 \
         }                                                                     \


/**
 * This Variadic functions iterate over a col major matrix and do operation
 * using one element only
 **/
#define BINARY_OP_SINGLE(dataA, dataB, dataOut, FUN, size) \
          for(auto i=0;i<size;i++){                                            \
                dataOut[i]=dataA[i] FUN dataB;                                 \
            }                                                                  \

#define RUN_BINARY_OP_SINGLE(dataA, dataB, dataOut, FUN, size) \
          if(FUN=="+")  {                                                      \
              BINARY_OP_SINGLE(dataA,dataB,dataOut,+,size)                     \
         }else if(FUN=="-")  {                                                 \
           BINARY_OP_SINGLE(dataA,dataB,dataOut,-,size)                        \
         }else if(FUN=="*")  {                                                 \
           BINARY_OP_SINGLE(dataA,dataB,dataOut,*,size)                        \
         }else if(FUN=="/")  {                                                 \
           BINARY_OP_SINGLE(dataA,dataB,dataOut,/,size)                        \
         }else if (FUN =="^"){                                                 \
              for(auto i=0;i<size;i++){                                        \
                dataOut[i]=std::pow(dataA[i], dataB);                          \
            }                                                                  \
         }else {                                                               \
             MPCR_API_EXCEPTION("Operation Not Supported", -1);                 \
         }                                                                     \


/**
 * This Variadic functions iterate over a row major matrix and do operation
 * using one element only
 **/
#define COMPARE_OP_SINGLE(dataA, dataB, dataOut, FUN, size) \
          for(auto i=0;i<size;i++){                                            \
               if(isnan(dataA[i]) || isnan(dataB) ){                           \
                dataOut[i]=INT_MIN;                                            \
            }else{                                                             \
                dataOut[ i ] =dataA[ i ] FUN dataB;                            \
            }                                                                  \
          }                                                                    \


#define COMPARE_OP(dataA, dataB, dataOut, FUN, sizeB, sizeA, sizeOut)          \
         for (auto i = 0; i < sizeOut; i++) {                                  \
            if(isnan(dataA[ i % sizeA ]) || isnan(dataB[ i % sizeB ]) ){       \
                dataOut[i]=INT_MIN;                                            \
            }else{                                                             \
                dataOut[ i ] =dataA[ i % sizeA ] FUN dataB[ i % sizeB ];       \
            }                                                                  \
            idx++;                                                             \
         }                                                                     \


/**
 * This dispatcher launch iterations using Row Major Representation for dataA
 * and a single Val for dataB (Comparisons)
 **/
#define RUN_COMPARE_OP_SINGLE(dataA, dataB, dataOut, FUN, sizeA)               \
         if(FUN==">")  {                                                       \
            COMPARE_OP_SINGLE(dataA,dataB,dataOut,>,sizeA)                     \
         }else if(FUN=="<")  {                                                 \
            COMPARE_OP_SINGLE(dataA,dataB,dataOut,<,sizeA)                     \
         }else if(FUN==">=")  {                                                \
            COMPARE_OP_SINGLE(dataA,dataB,dataOut,>=,sizeA)                    \
         }else if(FUN=="<=")  {                                                \
            COMPARE_OP_SINGLE(dataA,dataB,dataOut,<=,sizeA)                    \
         }else {                                                               \
             MPCR_API_EXCEPTION("Compare Operation Not Supported", -1);         \
         }                                                                     \


#define RUN_COMPARE_OP_SIMPLE(dataA, dataB, dataOut, FUN, sizeB, sizeA, sizeOut)\
         if(FUN==">")  {                                                       \
            COMPARE_OP(dataA,dataB,dataOut,>,sizeB,sizeA,sizeOut)              \
         }else if(FUN=="<")  {                                                 \
            COMPARE_OP(dataA,dataB,dataOut,<,sizeB,sizeA,sizeOut)              \
         }else if(FUN==">=")  {                                                \
            COMPARE_OP(dataA,dataB,dataOut,>=,sizeB,sizeA,sizeOut)             \
         }else if(FUN=="<=")  {                                                \
            COMPARE_OP(dataA,dataB,dataOut,<=,sizeB,sizeA,sizeOut)             \
         }else {                                                               \
             MPCR_API_EXCEPTION("Compare Operation Not Supported", -1);         \
         }                                                                     \

#endif //MPCR_BINARYOPERATIONSHELPER_HPP
