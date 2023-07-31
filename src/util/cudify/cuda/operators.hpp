#pragma once

namespace gpu {

template<typename type_t>
struct less_t : public std::binary_function<type_t, type_t, bool> {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a < b;
  }
};

template<typename type_t>
struct less_equal_t : public std::binary_function<type_t, type_t, bool> {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a <= b;
  }
};

template<typename type_t>
struct greater_t : public std::binary_function<type_t, type_t, bool> {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a > b;
  }
};

template<typename type_t>
struct greater_equal_t : public std::binary_function<type_t, type_t, bool> {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a >= b;
  }
};

template<typename type_t>
struct equal_to_t : public std::binary_function<type_t, type_t, bool> {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a == b;
  }
};

template<typename type_t>
struct not_equal_to_t : public std::binary_function<type_t, type_t, bool> {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a != b;
  }
};

template<typename type_t>
struct plus_t : public std::binary_function<type_t, type_t, type_t> {
	__forceinline__ __device__ __host__ type_t operator()(type_t a, type_t b) const {
    return a + b;
  }
};

template<typename type_t>
struct minus_t : public std::binary_function<type_t, type_t, type_t> {
	__forceinline__ __device__ __host__ type_t operator()(type_t a, type_t b) const {
    return a - b;
  }
};

template<typename type_t>
struct multiplies_t : public std::binary_function<type_t, type_t, type_t> {
  __forceinline__ __device__ __host__ type_t operator()(type_t a, type_t b) const {
    return a * b;
  }
};

template<typename type_t>
struct maximum_t  : public std::binary_function<type_t, type_t, type_t> {
  __forceinline__ __device__ __host__ type_t operator()(type_t a, type_t b) const {
    return (a < b) ? b : a;
  }
};

template<typename type_t>
struct minimum_t  : public std::binary_function<type_t, type_t, type_t> {
  __forceinline__ __device__ __host__ type_t operator()(type_t a, type_t b) const {
    return (b < a) ? b : a;
  }
};

}
