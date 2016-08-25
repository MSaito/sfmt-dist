#pragma once
#ifndef SFMT_DIST_ALIGNED_ALLOC_H
#define SFMT_DIST_ALIGNED_ALLOC_H
#include <cstddef>

namespace MersenneTwister {
    void * alignedAllocSub(size_t size);
    template<typename T>
        T alignedAlloc(size_t size) {
        return static_cast<T>(alignedAllocSub(size));
    }
    void alignedFree(void * addr);
}

#endif // SFMT_DIST_ALIGNED_ALLOC_H
