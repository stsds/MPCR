/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MMPR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPR_MATHEMATICALOPERATIONSHELPER_HPP
#define MPR_MATHEMATICALOPERATIONSHELPER_HPP

#include <utilities/MPRDispatcher.hpp>


#define RUN_OP(aOutput, size, __FUN__, ...)\
        for(auto i=0;i<size;i++){                                              \
            aOutput[i]=__FUN__(FIRST(__VA_ARGS__)REST(__VA_ARGS__)[i]);        \
        }                                                                      \



#define PERFORM_TRIG_OP(aInput, aOutput, aSize, OPERATION)\
        if(OPERATION=="cos"){                                                  \
        RUN_OP(aOutput,aSize,std::cos,aInput)                                  \
        }else if(OPERATION=="sin"){                                            \
        RUN_OP(aOutput,aSize,std::sin,aInput)                                  \
        }else if(OPERATION=="tan"){                                            \
        RUN_OP(aOutput,aSize,std::tan,aInput)                                  \
        }else if(OPERATION=="cosh"){                                           \
        RUN_OP(aOutput,aSize,std::cosh,aInput)                                 \
        }else if(OPERATION=="sinh"){                                           \
        RUN_OP(aOutput,aSize,std::sinh,aInput)                                 \
        }else if(OPERATION=="tanh"){                                           \
        RUN_OP(aOutput,aSize,std::tanh,aInput)                                 \
        }else{                                                                 \
        MPR_API_EXCEPTION("Unknown Trig Operation", -1);                       \
        }                                                                      \


#define PERFORM_INV_TRIG_OP(aInput, aOutput, aSize, OPERATION)\
        if(OPERATION=="acos"){                                                 \
        RUN_OP(aOutput,aSize,std::acos,aInput)                                 \
        }else if(OPERATION=="asin"){                                           \
        RUN_OP(aOutput,aSize,std::asin,aInput)                                 \
        }else if(OPERATION=="atan"){                                           \
        RUN_OP(aOutput,aSize,std::atan,aInput)                                 \
        }else if(OPERATION=="acosh"){                                          \
        RUN_OP(aOutput,aSize,std::acosh,aInput)                                \
        }else if(OPERATION=="asinh"){                                          \
        RUN_OP(aOutput,aSize,std::asinh,aInput)                                \
        }else if(OPERATION=="atanh"){                                          \
        RUN_OP(aOutput,aSize,std::atanh,aInput)                                \
        }else{                                                                 \
        MPR_API_EXCEPTION("Unknown Inverse Trig Operation", -1);               \
        }                                                                      \


#define PERFORM_ROUND_OP(aInput, aOutput, aSize, OPERATION)                    \
        if(OPERATION=="abs"){                                                  \
        RUN_OP(aOutput,aSize,std::fabs,aInput)                                 \
        }else if(OPERATION=="ceil"){                                           \
        RUN_OP(aOutput,aSize,std::ceil,aInput)                                 \
        }else if(OPERATION=="floor"){                                          \
        RUN_OP(aOutput,aSize,std::floor,aInput)                                \
        }else if(OPERATION=="trunc"){                                          \
        RUN_OP(aOutput,aSize,std::trunc,aInput)                                \
        }else{                                                                 \
        MPR_API_EXCEPTION("Unknown Round Operation", -1);                      \
        }                                                                      \


#endif //MPR_MATHEMATICALOPERATIONSHELPER_HPP
