#include "sfmt-dist.h"
#define _POSIX_C_SOURCE 200112L
#include <sfmt-dist/aligned_alloc.h>
#include <sfmt-dist/cpu_feature.h>
#include <stdlib.h>

namespace MersenneTwister {
    void * alignedAllocSub(size_t size)
    {
        void * result = NULL;
        cpu_feature_t cf = cpu_feature();
        int align = 8;
        if (cf.avx512f) {
            align = 64;
        } else if (cf.avx) {
            align = 32;
        } else {
            align = 16;
        }
        if (posix_memalign((void **)&result, align, size) != 0) {
            return NULL;
        }
        return result;
    }

    void alignedFree(void * addr)
    {
        free(addr);
    }
}
