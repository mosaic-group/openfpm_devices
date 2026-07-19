#pragma once

namespace gpu {

template<typename type_t>
struct less_t {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a < b;
  }
};

template<typename type_t>
struct less_equal_t {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a <= b;
  }
};

template<typename type_t>
struct greater_t {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a > b;
  }
};

template<typename type_t>
struct greater_equal_t {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a >= b;
  }
};

template<typename type_t>
struct equal_to_t {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a == b;
  }
};

template<typename type_t>
struct not_equal_to_t {
  __forceinline__ __device__ __host__ bool operator()(type_t a, type_t b) const {
    return a != b;
  }
};

template<typename type_t>
struct plus_t {
	__forceinline__ __device__ __host__ type_t operator()(type_t a, type_t b) const {
    return a + b;
  }

  __forceinline__ __device__ __host__ type_t reduceInitValue() const {
    return 0;
  }
};

template<typename type_t>
struct minus_t {
	__forceinline__ __device__ __host__ type_t operator()(type_t a, type_t b) const {
    return a - b;
  }

  __forceinline__ __device__ __host__ type_t reduceInitValue() const {
    return 0;
  }
};

template<typename type_t>
struct multiplies_t {
  __forceinline__ __device__ __host__ type_t operator()(type_t a, type_t b) const {
    return a * b;
  }
};

template<typename type_t>
struct maximum_t {
  __forceinline__ __device__ __host__ type_t operator()(type_t a, type_t b) const {
    return (a < b) ? b : a;
  }

  __forceinline__ __device__ __host__ type_t reduceInitValue() const {
    return std::numeric_limits<type_t>::min();
  }
};

template<typename type_t>
struct minimum_t {
  __forceinline__ __device__ __host__ type_t operator()(type_t a, type_t b) const {
    return (b < a) ? b : a;
  }

  __forceinline__ __device__ __host__ type_t reduceInitValue() const {
    return std::numeric_limits<type_t>::max();
  }
};

}
