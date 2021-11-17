#include <StandardSlabDefinitions.cuh>
#include <vector>
#include <Slab.cuh>
#include <cuda_profiler_api.h>
#include <unordered_map>

template<>
struct EMPTY<int *> {
    static constexpr int *value = nullptr;
};

int main() {
    const int size = 10000;
    std::hash<unsigned> hfn;
    
}