/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_BASICOPERATIONSHELPER_HPP
#define MPCR_BASICOPERATIONSHELPER_HPP

#include <math.h>


#define OPERATION_COL(dataA, dataB, dataOut, FUN, sizeB, accum)                \
                for(auto i=0;i<rows;i++){                                      \
                    for(auto j=0;j<cols;j++){                                  \
                        idx=(j*rows)+i;                                        \
                        dataOut[idx]=dataA[idx] FUN dataB[accum%sizeB];        \
                        accum++;                                               \
                    }                                                          \
                }                                                              \



#define OPERATION(dataA, dataB, dataOut, FUN, sizeB, accum)                    \
           for(auto i=0;i<size;i++){                                           \
                idx=idx%sizeB;                                                 \
                dataOut[i]=dataA[i] FUN dataB[idx];                            \
                idx+=accum + 1 ;                                               \
            }                                                                  \


#define RUN_OP(dataA, dataB, dataOut, FUN, sizeB, accum)                       \
         if(FUN=="+")  {                                                       \
              OPERATION(dataA,dataB,dataOut,+,sizeB,accum)                     \
         }else if(FUN=="-")  {                                                 \
           OPERATION(dataA,dataB,dataOut,-,sizeB,accum)                        \
         }else if(FUN=="*")  {                                                 \
           OPERATION(dataA,dataB,dataOut,*,sizeB,accum)                        \
         }else if(FUN=="/")  {                                                 \
           OPERATION(dataA,dataB,dataOut,/,sizeB,accum)                        \
         }else if(FUN=="^")  {                                                 \
           for(auto i=0;i<size;i++){                                           \
                idx=idx%sizeB;                                                 \
                dataOut[i]=std::pow(dataA[i],dataB[idx]);                      \
                idx+=accum + 1;                                                \
            }                                                                  \
         }else {                                                               \
             MPCR_API_EXCEPTION("Operation Not Supported", -1);                 \
         }                                                                     \



#define RUN_OP_COL(dataA, dataB, dataOut, FUN, sizeB, accum)                   \
         if(FUN=="+")  {                                                       \
              OPERATION_COL(dataA,dataB,dataOut,+,sizeB,accum)                 \
         }else if(FUN=="-")  {                                                 \
           OPERATION_COL(dataA,dataB,dataOut,-,sizeB,accum)                    \
         }else if(FUN=="*")  {                                                 \
           OPERATION_COL(dataA,dataB,dataOut,*,sizeB,accum)                    \
         }else if(FUN=="/")  {                                                 \
           OPERATION_COL(dataA,dataB,dataOut,/,sizeB,accum)                    \
         }else if(FUN=="^")  {                                                 \
                for(auto i=0;i<rows;i++){                                      \
                    for(auto j=0;j<cols;j++){                                  \
                        idx=(j*rows)+i;                                        \
                        dataOut[idx]=std::pow(dataA[idx],dataB[accum%sizeB]);  \
                        accum++;                                               \
                    }                                                          \
                }                                                              \
         }else {                                                               \
             MPCR_API_EXCEPTION("Operation Not Supported", -1);                 \
         }                                                                     \

#endif //MPCR_BASICOPERATIONSHELPER_HPP
