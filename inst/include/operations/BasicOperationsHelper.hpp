

#ifndef MPR_BASICOPERATIONSHELPER_HPP
#define MPR_BASICOPERATIONSHELPER_HPP


#define OPERATION(dataA, dataB, dataOut, FUN, val, accum)                      \
           for(auto i=0;i<size;i++){                                           \
                idx=idx%stat_size;                                             \
                dataOut[i]=dataA[i] FUN dataB[idx];                            \
                idx+=accum + 1;                                                \
            }


#define RUN_OP(dataA, dataB, dataOut, FUN, val, accum)                         \
         if(FUN=="+")  {                                                       \
              OPERATION(dataA,dataB,dataOut,+,val,accum)                       \
         }else if(FUN=="-")  {                                                 \
           OPERATION(dataA,dataB,dataOut,-,val,accum)                          \
         }else if(FUN=="*")  {                                                 \
           OPERATION(dataA,dataB,dataOut,*,val,accum)                          \
         }else if(FUN=="/")  {                                                 \
           OPERATION(dataA,dataB,dataOut,/,val,accum)                          \
         }else {                                                               \
             MPR_API_EXCEPTION("Operation Not Supported", -1);                 \
         }                                                                     \



#endif //MPR_BASICOPERATIONSHELPER_HPP
