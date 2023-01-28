

#ifndef MPR_BINARYOPERATIONSHELPER_HPP
#define MPR_BINARYOPERATIONSHELPER_HPP

#include <limits.h>


/************************** Operations *******************************/

/**
 * This Variadic functions iterate over a row major matrix in a column major form
 * accumulate is a normal iterator over dataB
 **/
#define BINARY_OP(dataA, dataB, dataOut, FUN, sizeB, accum)                    \
         for (auto i = 0; i < col; i++) {                                      \
                size_t start_idx = 0;                                          \
                for (auto j = 0; j < row; j++) {                               \
                   start_idx = i + ( j * col );                                \
                   dataOut[ start_idx ] =dataA[ start_idx ] FUN                \
                   dataB[ accum % sizeB ];                                     \
                   idx++;                                                      \
                }                                                              \
         }                                                                     \

/**
 * This Variadic functions iterate over a row major matrix and do operation
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
             MPR_API_EXCEPTION("Operation Not Supported", -1);                 \
         }                                                                     \

/**
 * This dispatcher launch iterations using Column representation for dataA
 * and Normal Iteration for dataB (Operations)
 **/
#define RUN_BINARY_OP(dataA, dataB, dataOut, FUN, sizeB, accum)                \
         if(FUN=="+")  {                                                       \
              BINARY_OP(dataA,dataB,dataOut,+,sizeB,accum)                     \
         }else if(FUN=="-")  {                                                 \
           BINARY_OP(dataA,dataB,dataOut,-,sizeB,accum)                        \
         }else if(FUN=="*")  {                                                 \
           BINARY_OP(dataA,dataB,dataOut,*,sizeB,accum)                        \
         }else if(FUN=="/")  {                                                 \
           BINARY_OP(dataA,dataB,dataOut,/,sizeB,accum)                        \
         }else if(FUN=="^")  {                                                 \
            for (auto i = 0; i < col; i++) {                                   \
                size_t start_idx = 0;                                          \
                for (auto j = 0; j < row; j++) {                               \
                   start_idx = i + ( j * col );                                \
                   dataOut[ start_idx ]=std::pow(dataA[ start_idx ],           \
                   dataB[ accum % sizeB ]);                                    \
                   idx++;                                                      \
                }                                                              \
             }                                                                 \
         }else {                                                               \
             MPR_API_EXCEPTION("Operation Not Supported", -1);                 \
         }                                                                     \

/************************** Comparisons *******************************/

/**
 * This Variadic functions iterate over a row major matrix in a column major form
 * accumulate is a normal iterator over dataB
 **/
#define COMPARE_OP(dataA, dataB, dataOut, FUN, sizeB, accum)                   \
         for (auto i = 0; i < col; i++) {                                      \
                size_t start_idx = 0;                                          \
                for (auto j = 0; j < row; j++) {                               \
                   start_idx = i + ( j * col );                                \
                   if(isnan(dataA[start_idx]) || isnan(dataB[accum%sizeB]) ){   \
                        dataOut[start_idx]   = INT_MIN;                        \
                   }else{                                                      \
                      dataOut[ start_idx ] =dataA[ start_idx ] FUN             \
                      dataB[ accum % sizeB ];                                  \
                   }                                                           \
                   idx++;                                                      \
                }                                                              \
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


#define COMPARE_OP_SIMPLE(dataA, dataB, dataOut, FUN, sizeB, sizeA)            \
         for (auto i = 0; i < sizeA; i++) {                                    \
            if(isnan(dataA[i]) || isnan(dataB[i]) ){                           \
                dataOut[i]=INT_MIN;                                            \
            }else{                                                             \
                dataOut[ i ] =dataA[ i ] FUN dataB[ i % sizeB ];               \
            }                                                                  \
            idx++;                                                             \
         }                                                                     \
/**
 * This dispatcher launch iterations using Column representation for dataA
 * and Normal Iteration for dataB (Comparisons)
 **/
#define RUN_COMPARE_OP(dataA, dataB, dataOut, FUN, sizeB, accum)               \
         if(FUN==">")  {                                                       \
            COMPARE_OP(dataA,dataB,dataOut,>,sizeB,accum)                      \
         }else if(FUN=="<")  {                                                 \
            COMPARE_OP(dataA,dataB,dataOut,<,sizeB,accum)                      \
         }else if(FUN==">=")  {                                                \
            COMPARE_OP(dataA,dataB,dataOut,>=,sizeB,accum)                     \
         }else if(FUN=="<=")  {                                                \
            COMPARE_OP(dataA,dataB,dataOut,<=,sizeB,accum)                     \
         }else {                                                               \
             MPR_API_EXCEPTION("Compare Operation Not Supported", -1);         \
         }                                                                     \

/**
 * This dispatcher launch iterations using Row Major Representation for dataA
 * and a single Val for dataB (Comparisons)
 **/
#define RUN_COMPARE_OP_SINGLE(dataA, dataB, dataOut, FUN, sizeA) \
         if(FUN==">")  {                                                       \
            COMPARE_OP_SINGLE(dataA,dataB,dataOut,>,sizeA)                     \
         }else if(FUN=="<")  {                                                 \
            COMPARE_OP_SINGLE(dataA,dataB,dataOut,<,sizeA)                     \
         }else if(FUN==">=")  {                                                \
            COMPARE_OP_SINGLE(dataA,dataB,dataOut,>=,sizeA)                    \
         }else if(FUN=="<=")  {                                                \
            COMPARE_OP_SINGLE(dataA,dataB,dataOut,<=,sizeA)                    \
         }else {                                                               \
             MPR_API_EXCEPTION("Compare Operation Not Supported", -1);         \
         }                                                       \


#define RUN_COMPARE_OP_SIMPLE(dataA, dataB, dataOut, FUN, sizeB, sizeA)        \
         if(FUN==">")  {                                                       \
            COMPARE_OP_SIMPLE(dataA,dataB,dataOut,>,sizeB,sizeA)               \
         }else if(FUN=="<")  {                                                 \
            COMPARE_OP_SIMPLE(dataA,dataB,dataOut,<,sizeB,sizeA)               \
         }else if(FUN==">=")  {                                                \
            COMPARE_OP_SIMPLE(dataA,dataB,dataOut,>=,sizeB,sizeA)              \
         }else if(FUN=="<=")  {                                                \
            COMPARE_OP_SIMPLE(dataA,dataB,dataOut,<=,sizeB,sizeA)              \
         }else {                                                               \
             MPR_API_EXCEPTION("Compare Operation Not Supported", -1);         \
         }                                                                     \

#endif //MPR_BINARYOPERATIONSHELPER_HPP
