#include <Operations.cuh>
#include <functional>

#ifndef LSKIPVECTOR_STANDARDSLABS_CUH
#define LSKIPVECTOR_STANDARDSLABS_CUH

struct data_t {

};

class Data_tDeleter {

};

template<>
struct EMPTY<data_t *> {
    static constexpr data_t *value = nullptr;
};


template<>
__forceinline__ __device__ unsigned compare(data_t *const &lhs, data_t *const &rhs) {
    // TODO: complete this
    return 1;
}


namespace std {
    template<>
    struct std::hash<data_t *> {
        std::size_t operator()(data_t *&x) {
            return std::hash<std::string>{}(x->data) ^ std::hash<std::size_t>{}(x->size);
        }
    };
}

template<>
struct EMPTY<unsigned> {
    static const unsigned value = 0;
}

template<>
__forceinline__ __device__ unsigned compare(const unsigned &lhs, const unsigned &rhs) {
    return lhs - rhs;
}

template<> 
__forceinline__ __device__ unsigned compare(const unsigned long long &lhs, const unsigned long long &rhs) {
    return lhs - rhs;
}

#endif