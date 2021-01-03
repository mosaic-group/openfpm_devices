#ifndef CUDIFY_HARDWARE_COMMON_HPP_
#define CUDIFY_HARDWARE_COMMON_HPP_


#include <initializer_list>

#ifdef CUDA_ON_CPU

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

    constexpr dim3(const dim3 & d) : x(d.x), y(d.y), z(d.z) {}

    template<typename T>
    constexpr dim3(const std::initializer_list<T> & list) 
    {
        auto it = list.begin();

        x = *it;
        ++it;
        y = *it;
        ++it;
        z = *it;
    }
};

#endif

#endif