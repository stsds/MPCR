

#ifndef MPR_BASICOPERATIONSHELPER_HPP
#define MPR_BASICOPERATIONSHELPER_HPP

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
             MPR_API_EXCEPTION("Operation Not Supported", -1);                 \
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
             MPR_API_EXCEPTION("Operation Not Supported", -1);                 \
         }                                                                     \

#endif //MPR_BASICOPERATIONSHELPER_HPP
