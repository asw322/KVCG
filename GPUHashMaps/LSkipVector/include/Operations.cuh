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
    typedef unsigned long long KSub;

    unsigned level;

    Node *right;
    Node *down;

    KSub max;
    KSub min;

    SlabData<K, V> *slab;
};

template<typename K, typename V>
Node::Node(K k, V v, int level) {
    this->right = new Node*[level+1];
    this->slab = new SlabData<K, V>(k, v);
}


// TODO: what does s.keyValue return?
template<typename K, typename V>
std::ostream &operator<<(std::ostream &output, const SlabData<K, V> &s) {
    output << s.keyValue;
    return output;
}

template<typename K, typename V>
struct NodeCtx {
    // default constructor
    NodeCtx() : nodes(nullptr), num_of_buckets(0) {}

    volatile Node<K, V> **nodes;
    unsigned levels;
    unsigned num_of_buckets;
};



template<typename K, typename V>
__forceinline__ __device__ void LockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &landId, const unsigned &level, volatile NodeCtx<K, V> nodeCtx) {
    if(landId == 0) {
        auto ilock = (int *) &(nodeCtx.nodes[level][src_bucket]->slab->ilock);
        while(atomicCAS(ilock, 0, -1) != 0);
    }

    __syncwarp();
}

template<typename K, typename V>
__forceinline__ __device__ void UnlockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &landId, const unsigned &level, volatile NodeCtx<K, V> nodeCtx) {
    if(landId == 0) {
        auto ilock = (int *) &(nodeCtx.nodes[level][src_bucket]->slab->ilock);
        atomicExch(ilock, 0);
    }
}

template<typename K, typename V>
__forceinline__ __device__ void SharedLockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &laneId, const unsigned &level volatile NodeCtx<K, V> nodeCtx) {
    if(laneId == 0) {
        auto ilock = (int *) &(nodeCtx.nodes[level][src_bucket]->slab->ilock);
        while(true) {
            auto pred = *ilock;
            if(pred != 1 && atomicCAS(ilock, pred, pred + 1) == pred) {
                break;
            }
        }
    }

    __syncwarp();
}

template<typename K, typename V>
__forceinline__ __device__ typename SlabData<K, V>::KSub ReadSlabKey(const unsigned long long &next, const unsigned &src_bucket, const unsigned laneId, const unsigned level, volatile NodeCtx<K, V> nodeCtx) {
    static_assert(sizeof(typname SlabData<K, V>::Ksub) >= sizeof(void*), "Need to be able to substitute pointers for values");
    return next == BASE_SLAB ? nodeCtx.nodes[level][src_bucket]->slab->key[laneId] : ((SlabData<K, V> *) next)->key[laneId];
}

template<typename K, typename V>
__forceinline__ __device__ V ReadSlabValue(const unsigned long long &next, const unsigned &src_bucket, const unsigned laneId, const unsigned level, volatile NodeCtx<K, V> nodeCtx) {
    return (next == BASE_SLAB ? nodeCtx.nodes[level][src_bucket]->slab->value[laneId] : ((SlabData<K, V> *) next)->value[laneId]);
}


template<typename K, typename V> 
__forceinline__ __device__ void operation_search(bool &is_active, const K &myKey, V &myValue, const unsigned &modhash, volatile NodeCtx<K, V> nodeCtx, unsigned num_of_buckets, Node<K, V> *__restrict__ curr_node) {
    if(curr_node.max < myKey) {
        operation_search(is_active, myKey, myValue, modhash, nodeCtx, num_of_buckets, curr_node.right);
    } 
    else if(curr_node.max > myKey && curr_node.min < myKey) {
        if(curr_node.level > 0) {
            operation_search(is_active, myKey, myValue, modhash, nodeCtx, num_of_buckets, curr_node.down);
        }

        if(curr_node.level == 0) {
            for(int i = 0; i < curr_node->slab.key.length; i++) {
                auto masked_ballot = (unsigned) (__ballot_sync(~0u, compare((K) curr_node->slab.key[i], (K) myKey) == 0) 
            }
        }
    }

    // const unsigned laneId = threadIdx.x & 0x1Fu;
    // unsigned long long next = BASE_SLAB;
    
    // unsigned work_queue = __ballot_sync(~0u, is_active);
    // unsigned last_work_queue = work_queue;

    // const auto threadKey = (unsigned long long) myKey; 

    // while(work_queue != 0) {
    //     next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;

    //     unsigned src_lane = __ffs((int) work_queue) - 1;
    //     unsigned long long src_key = __shfl_sync(~0u, threadKey, (int) src_lane);
    //     unsigned src_bucket = __shfl_sync(~0u, modhash, (int) src_lane);
    //     SharedLockSlab(next, src_bucket, laneId, nodeCtx);


    //     unsigned long long read_key = (unsigned long long) ReadSlabKey(next, src_bucket, laneId, slabs);


    //     if(FOUND_VALUE) {
    //         V read_value = ReadSlabValue(next, src_bucket, laneId, nodeCtx.levels, nodeCtx);
            
    //         auto found_lane = (unsigned) (__ffs(masked_ballot) - 1);
    //         auto found_value = (V) __shfl_sync(~0u, (unsigned long long) read_value, found_lane);
    //         if(laneId == src_lane) {
    //             myValue = found_value;
    //             is_active = false;
    //         }
    //     }

    //     last_work_queue = work_queue;

    //     work_queue = __ballot_sync(~0u, is_active);

    //     SharedUnlockSlab(next, src_bucket, laneId, slabs);
    // }
}



template<typename K, typename V>
__forceinline__ __device__ void operation_delete(bool &is_active, const K &myKey, V &myValue, const unsigned &modhash, volatile SlabData<K, V> **_restrict__ slabs, unsigned num_of_buckets) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;

    unsigned work_queue = __ballot_sync(~0u, is_active);
    unsigned last_work_queue = work_queue;

    while(work_queue != 0) {
        auto src_lane = (unsigned) (__ffs((int) work_queue) -1);
        auto src_key = (K) __shfl_sync(~0u, (unsigned long long) myKey, src_lane);
        unsigned src_bucket = __shfl_sync(~0u, modhash, (int) src_lane);
        LockSlab(next, src_bucket, landId, slabs);

        // perform the delete
        K read_key = ReadSlabKey(next, src_bucket, landId, slabs);
        auto mask_ballot = (unsigned) (__ballot_sync(~0u, compare(read_key, src_key) == 0) & VALID_KEY_MASK);

        // TODO: need to implement
        if(FOUND_VALUE) {
            unsigned long long next_ptr = __shfl_sync(~0u, (unsigned long long) read_key, ADDRESS_LANE -1);
            if(next_ptr == 0) {
                is_active = false;
                myValue = EMPTY<V>::value;
            } else {
                UnlockSlab(next, src_bucket, laneId, slabs);
                __syncwarp();
                next = next_ptr;
            }
        }

        UnlockSlab(next, src_bucket, landId, slabs);
    }
}


template<typename K, typename V> 
__forceinline__ __device__ void operation_replace(bool &is_active, const K &myKey, V &myValue, const unsigned &modhash, volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets, WarpAllocCtx<K, V> ctx) {

}



template<typename K, typename V>
NodeCtx<K, V> *setUpGroup(groupallocator::GroupAllocator &gAlloc, unsigned size, int gpuid = 0, cudaStream_t stream = cudaStreamDefault) {
    gpuErrchk(cudaSetDevice(gpuid));

    auto ndtx = new NodeCtx<K, V>();

    ndtx->num_of_buckets = size;
    std::cerr << "Size of index is " << size << std::endl;
    std::cerr << "Each slab is " << sizeof(SlabData<K, V>) << "B" << std::endl;

    gAlloc.allocate(&(ndtx->slabs), sizeof(void *) * ndtx->num_of_buckets, false);

    for(int i = 0; i < ndtx->num_of_buckets; i++) {
        gAlloc.allocate(&(ndex->slabs[i], sizeof(SlabData<K, V>), false));
        static_assert(sizeof(ndtx->slabs[i]->key[0]) >= sizeof(void *), "The key size needs to be greater or equal to the size of a memory address");
        memset((void *) (ndtx->slabs[i]), 0, sizeof(SlabData<K, V>));

        for(int j = 0; j < 31; j++) {
            ndtx->slabs[i]->key[j] = EMPTY<K>::value;
        }
    }

    gAlloc.moveToDevice(gpuid, stream);

    return ndtx;
}


#endif
