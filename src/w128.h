#pragma once
#ifndef SFMT_DIST_W128_H
#define SFMT_DIST_W128_H

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
    union w128_t {
        uint64_t u64[2];
        uint32_t u32[4];
        double   d[2];
#if HAVE_IMMINTRIN_H
        __m128i   si128;
        __m128d   sd128;
#endif
    };

    union w128x32_t {
        uint32_t u32[4];
#if HAVE_IMMINTRIN_H
        __m128i  si128;
#endif
    };
    union w128xd_t {
        double d[2];
#if HAVE_IMMINTRIN_H
        __m128d sd128;
#endif
    };
}

#endif // SFMT_DIST_W128_H
