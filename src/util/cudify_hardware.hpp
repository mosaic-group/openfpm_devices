#ifndef CUDIFY_HARDWARE_HPP_
#define CUDIFY_HARDWARE_HPP_

//#define ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
/*#define ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED
#define ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
#define ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#define ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#define ALPAKA_ACC_ANY_BT_OACC_ENABLED
#define ALPAKA_ACC_ANY_BT_OMP5_ENABLED*/


#include <alpaka/alpaka.hpp>



using Dim_alpa = alpaka::DimInt<3>;
using Idx_alpa = std::size_t;

using Acc_alpa = alpaka::AccCpuFibers<Dim_alpa, Idx_alpa>;

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

struct uint3
{
    unsigned int x, y, z;
};

struct dim3
{
    unsigned int x, y, z;

    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    constexpr operator uint3(void) const { return uint3{x, y, z}; }
};

#endif