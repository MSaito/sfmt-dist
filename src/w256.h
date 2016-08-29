#pragma once
#ifndef SFMT_DIST_W256_H
#define SFMT_DIST_W256_H

#include "sfmt-dist.h"
#if HAVE_MEMORY_H
#include <memory.h>
#endif
#if HAVE_STRING_H
#include <string.h>
#endif

#if HAVE_ZMMINTRIN_H
#include <zmmintrin.h>
#endif

#if HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif

namespace MersenneTwister {
    union w256_t {
        uint64_t u64[4];
        uint32_t u32[8];
        double   d[4];
#if HAVE_IMMINTRIN_H
        __m256i   si256;
        __m256d   sd256;
#endif
    };

    union w256x32_t {
        uint32_t u32[8];
#if HAVE_IMMINTRIN_H
        __m256i  si256;
#endif
    };
    union w256xd_t {
        double d[4];
#if HAVE_IMMINTRIN_H
        __m256d sd256;
#endif
    };
}

#endif // SFMT_DIST_W256_H
