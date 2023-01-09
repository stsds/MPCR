

#ifndef MPR_BASICOPERATIONSHELPER_HPP
#define MPR_BASICOPERATIONSHELPER_HPP



#define RUN_OP(dataA, dataB, dataOut, FUN, val)                                \
         if(FUN=="+")  {                                                       \
            for(auto i=0;i<size;i++){                                          \
                idx=i%val;                                                     \
                idx=idx%size;                                                  \
                dataOut[i]=dataA[i] + dataB[idx];                              \
            }                                                                  \
         }else if(FUN=="-")  {                                                 \
           for(auto i=0;i<size;i++){                                           \
                idx=i%val;                                                     \
                idx=idx%size;                                                  \
                dataOut[i]=dataA[i] - dataB[idx];                              \
            }                                                                  \
         }else if(FUN=="*")  {                                                 \
           for(auto i=0;i<size;i++){                                           \
                idx=i%val;                                                     \
                idx=idx%size;                                                  \
                dataOut[i]=dataA[i] * dataB[idx];                              \
            }                                                                  \
         }else if(FUN=="/")  {                                                 \
           for(auto i=0;i<size;i++){                                           \
                idx=i%val;                                                     \
                idx=idx%size;                                                  \
                dataOut[i]=dataA[i] / dataB[idx];                              \
            }                                                                  \
         }else {                                                               \
             MPR_API_EXCEPTION("Operation Not Supported", -1);                 \
         }                                                                     \




#endif //MPR_BASICOPERATIONSHELPER_HPP
