// TODO: CREATE CMAKE FILE TO COMPILE THIS. 


#include <groupallocator>
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <gpuErrchk.cuh>
#include <gpumemory.cuh>
#include <iostream>
#include <nvfunctional>
#include "ImportantDefinitions.cuh"

#ifndef OPERATIONS_CUH
#define OPERATIONS_CUH

const unsigned long long EMPTY_POINTER = 0;
#define BASE_SLAB 0



template<typename K, typename V>
struct SlabData {
    typedef K KSub;

    // union total of 128 bytes
    union {
        int ilock;
        char p[128];
    };

    // 32nd key element points to the head address of the next SlabData key block
    // key[32] :: unsigned long long *next
    KSub key[32];
    V value[32];
};

template<typename V>
struct SlabData<char, V> {
    typedef unsigned long long KSub;

    union {
        int ilock;
        char p[128];
    };

    KSub key[32];
    V value[32];
};

template<typename V> 
struct SlabData<short, V> {
    typedef unsigned long long KSub;
    union {
        int ilock;
        char p[128];
    };

    KSub key[32];
    V value[32];
};

template<typename V>
struct SlabData<unsigned, V> {
    typedef unsigned long long KSub;
    union {
        int ilock;
        char p[128];
    };

    KSub key[32];
    V value[32];
};





template<typename K, typename V>
struct Node {
    Node **forward;
    SlabData<K, V> *slab;
};

template<typename K, typename V>
Node::Node(K k, V v, int level) {
    forward = new Node*[level+1];
    this->slab = new SlabData<K, V>(k, v);
}


// TODO: what does s.keyValue return?
template<typename K, typename V>
std::ostream &operator<<(std::ostream &output, const SlabData<K, V> &s) {
    output << s.keyValue;
    return output;
}

template<typename K, typename V>
void setUpGroup(groupallocator::GroupAllocator &gAlloc, unsigned size, int gpuid = 0, cudaStream_t stream = cudaStreamDefault) {
    gpuErrchk(cudaSetDevice(gpuid));
    return;
}


#endif


int main() {
    std::cout << "hello" << std::endl;
}