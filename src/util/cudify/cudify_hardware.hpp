#ifndef CUDIFY_ALPAKA_HARDWARE_HPP_
#define CUDIFY_ALPAKA_HARDWARE_HPP_

//#define ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#define ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
#define ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
//#define ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
//#define ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
//#define ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
//#define ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
//#define ALPAKA_ACC_ANY_BT_OACC_ENABLED
//#define ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#include "cudify_hardware_common.hpp"
#include <alpaka/alpaka.hpp>



using Dim_alpa = alpaka::DimInt<3>;
using Idx_alpa = std::size_t;

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
using Acc_alpa = alpaka::AccCpuFibers<Dim_alpa, Idx_alpa>;
#elif defined(ALPAKA_ACC_ANY_BT_OMP5_ENABLED)
using Acc_alpa = alpaka::AccOmp5<Dim_alpa, Idx_alpa>;
#endif

using QueueProperty_alpa = alpaka::Blocking;
using Queue_alpa = alpaka::Queue<Acc_alpa, QueueProperty_alpa>;

using Vec_alpa = alpaka::Vec<Dim_alpa, Idx_alpa>;

using WorkDiv_alpa = alpaka::WorkDivMembers<Dim_alpa, Idx_alpa>;

typedef decltype(alpaka::getDevByIdx<Acc_alpa>(0u)) AccType_alpa;

struct alpa_base_structs
{
    AccType_alpa * devAcc;
    Queue_alpa * queue;

    const Acc_alpa * accKer;
    bool initialized = false;

};


#endif