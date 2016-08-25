/**
 * @file mt19937_64.cpp
 *
 *\japanese
 * @brief MT19937-64 64ビット疑似乱数生成器
 * いまのところ速い
 *
 * この疑似乱数生成器の周期は2<sup>19937</sup>-1である。
 *\endjapanese
 *
 *\english
 * @brief 64 bit pesudo-random number generator MT19937-64
 *\endenglish
 *
 * @author Mutsuo Saito (Manieth Corp.)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * Copyright (C) 2015 Mutsuo Saito, Makoto Matsumoto Manieth Corp. and
 * Hiroshima University.
 * All rights reserved.
 *
 * The 3-clause BSD License is applied to this software, see
 * COPYING.
 */
#include "sfmt-dist.h"
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdexcept>
#include <cstdio>
#include <sfmt-dist/mt19937_64.h>
#include <sfmt-dist/cpu_feature.h>
#include <sfmt-dist/aligned_alloc.h>
#if HAVE_EMMINTRIN_H
#include <emmintrin.h>
#endif
#if HAVE_TMMINTRIN_H
#include <tmmintrin.h>
#endif
#if HAVE_NMMINTRIN_H
#include <nmmintrin.h>
#endif
#if HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif
#if HAVE_ZMMINTRIN_H
#include <zmmintrin.h>
#endif
#if HAVE_X86INTRIN_H
#include <x86intrin.h>
#endif

/*
 * HAVE_XXXX で Intel の SIMD を指定しているのは、コンパイル時にその
 * 機能を使ったコード生成ができるかどうかということを示している。
 * 実行時に機能が使えるかどうかは、cpu_feature() 関数によって判定し、
 * 実際に使える機能を使う。
 */

#define MATA UINT64_C(0xB5026F5AA96619E9)
#define UPPER_MASK UINT64_C(0xFFFFFFFF80000000)
#define LOWER_MASK UINT64_C(0x7FFFFFFF)
#define TEMPER_MASK1 UINT64_C(0x5555555555555555)
#define TEMPER_MASK2 UINT64_C(0x71D67FFFEDA60000)
#define TEMPER_MASK3 UINT64_C(0xFFF7EEE000000000)
#define TEMPER_SH1 29
#define TEMPER_SH2 17
#define TEMPER_SH3 37
#define TEMPER_SH4 43
#define POS 156

namespace {
    using namespace std;
    //const uint64_t mata = MATA;
    const int pos = POS;
    //const int mexp = 19937;
    const int size = 312;
    //const uint64_t upper_mask = UPPER_MASK;
    //const uint64_t lower_mask = LOWER_MASK;

#if HAVE_AVX512F
    union w512_t {
        uint64_t u64[8];
        __m512i simd512;
    };
    const w512_t mata512 = {{MATA, MATA, MATA, MATA, MATA, MATA, MATA, MATA}};
    const w512_t ones512 = {{1, 1, 1, 1, 1, 1, 1, 1}};
    const w512_t um512 = {{UPPER_MASK, UPPER_MASK, UPPER_MASK, UPPER_MASK,
                           UPPER_MASK, UPPER_MASK, UPPER_MASK, UPPER_MASK}};
    const w512_t lm512 = {{LOWER_MASK, LOWER_MASK, LOWER_MASK, LOWER_MASK,
                           LOWER_MASK, LOWER_MASK, LOWER_MASK, LOWER_MASK}};
    const w512_t tm1_512 = {{TEMPER_MASK1, TEMPER_MASK1,
                             TEMPER_MASK1, TEMPER_MASK1,
                             TEMPER_MASK1, TEMPER_MASK1,
                             TEMPER_MASK1, TEMPER_MASK1}};
    const w512_t tm2_512 = {{TEMPER_MASK2, TEMPER_MASK2,
                             TEMPER_MASK2, TEMPER_MASK2,
                             TEMPER_MASK2, TEMPER_MASK2,
                             TEMPER_MASK2, TEMPER_MASK2}};
    const w512_t tm3_512 = {{TEMPER_MASK3, TEMPER_MASK3,
                             TEMPER_MASK3, TEMPER_MASK3,
                             TEMPER_MASK3, TEMPER_MASK3,
                             TEMPER_MASK3, TEMPER_MASK3}};
#endif

#if HAVE_AVX2
    union w256_t {
        uint64_t u64[4];
        __m256i simd256;
    };
    const w256_t mata256 = {{MATA, MATA, MATA, MATA}};
    const w256_t ones256 = {{1, 1, 1, 1}};
    const w256_t um256 = {{UPPER_MASK, UPPER_MASK, UPPER_MASK, UPPER_MASK}};
    const w256_t lm256 = {{LOWER_MASK, LOWER_MASK, LOWER_MASK, LOWER_MASK}};
    const w256_t tm1_256 = {{TEMPER_MASK1, TEMPER_MASK1,
                             TEMPER_MASK1, TEMPER_MASK1}};
    const w256_t tm2_256 = {{TEMPER_MASK2, TEMPER_MASK2,
                             TEMPER_MASK2, TEMPER_MASK2}};
    const w256_t tm3_256 = {{TEMPER_MASK3, TEMPER_MASK3,
                             TEMPER_MASK3, TEMPER_MASK3}};
#endif

#if HAVE_SSSE3
    union w128_t {
        uint64_t u64[2];
        __m128i simd128;
    };
    const w128_t mata128 = {{MATA, MATA}};
    const w128_t ones128 = {{1, 1}};
    const w128_t um128 = {{UPPER_MASK, UPPER_MASK}};
    const w128_t lm128 = {{LOWER_MASK, LOWER_MASK}};
    const w128_t tm1_128 = {{TEMPER_MASK1, TEMPER_MASK1}};
    const w128_t tm2_128 = {{TEMPER_MASK2, TEMPER_MASK2}};
    const w128_t tm3_128 = {{TEMPER_MASK3, TEMPER_MASK3}};
#endif

#if defined(NOT_USE_MAG_ARRAY)
    const uint64_t mata64 = MATA;
#else
    const uint64_t mag[2] = {0, MATA};
#endif

#if HAVE_AVX512F
    /*
     * AVX512F による512bit SIMD 用の load
     */
    static inline __m512i load512(uint64_t * addr) {
        return _mm512_load_si512((__m512i *)addr);
    }

    /*
     * AVX512F による512bit SIMD 用の整列されていないアドレスからのload
     */
    static inline __m512i loadu512(uint64_t * addr) {
        return _mm512_loadu_si512((__m512i *)addr);
    }

    /*
     * AVX512F による512bit SIMD 用のstore
     */
    static inline void store512(uint64_t * addr, __m512i x) {
        _mm512_store_si512((__m512i *)addr, x);
    }

    /*
     * AVX512F による512bit SIMD 用の漸化式関数
     */
    static inline __m512i recursion512(__m512i x, __m512i y, __m512i z) {
        x = _mm512_and_si512(x, um512.simd512);
        y = _mm512_and_si512(y, lm512.simd512);
        x = _mm512_or_si512(x, y);
        y = _mm512_srli_epi64(x, 1);
#if 0
        x = _mm512_and_si512(x, ones512.simd512);
        x = _mm512_cmpeq_epi64(x, ones512.simd512);
        x = _mm512_and_si512(x, mata512.simd512);
#else
        __mmask8 flag = _mm512_test_epi64_mask(x, ones512.simd512);
        __m512i x1 = _mm512_maskz_mov_epi64(flag, mata512.simd512);
#endif
        z = _mm512_xor_si512(z, y);
        z = _mm512_xor_si512(z, x1);
        return z;
    }

    /*
     * 512bit 用の出力変換
     */
    static inline void output_conv512(uint64_t array[8]) {
        __m512i x = load512(array);
        __m512i y = _mm512_srli_epi64(x, TEMPER_SH1);
        y = _mm512_and_si512(y, tm1_512.simd512);
        x = _mm512_xor_si512(x, y);
        y = _mm512_slli_epi64(x, TEMPER_SH2);
        y = _mm512_and_si512(y, tm2_512.simd512);
        x = _mm512_xor_si512(x, y);
        y = _mm512_slli_epi64(x, TEMPER_SH3);
        y = _mm512_and_si512(y, tm3_512.simd512);
        x = _mm512_xor_si512(x, y);
        y = _mm512_srli_epi64(x, TEMPER_SH4);
        x = _mm512_xor_si512(x, y);
        store512(array, x);
    }
#endif

#if HAVE_AVX2
    /*
     * AVX2 による256bit SIMD 用の load
     */
    static inline __m256i load256(uint64_t * addr) {
        return _mm256_load_si256((__m256i *)addr);
    }

    /*
     * AVX2 による256bit SIMD 用の整列されていないアドレスからのload
     */
    static inline __m256i loadu256(uint64_t * addr) {
        //return _mm256_loadu_si256((__m256i *)addr);
        return _mm256_lddqu_si256((__m256i *)addr);
    }

    /*
     * AVX2 による256bit SIMD 用のstore
     */
    static inline void store256(uint64_t * addr, __m256i x)
    {
        _mm256_store_si256((__m256i *)addr, x);
    }

    /*
     * AVX2 による256bit SIMD 用の漸化式関数
     */
    static inline __m256i recursion256(__m256i x, __m256i y, __m256i z)
    {
        x = _mm256_and_si256(x, um256.simd256);
        y = _mm256_and_si256(y, lm256.simd256);
        x = _mm256_or_si256(x, y);
        y = _mm256_srli_epi64(x, 1);
        x = _mm256_and_si256(x, ones256.simd256);
        x = _mm256_cmpeq_epi64(x, ones256.simd256);
        x = _mm256_and_si256(x, mata256.simd256);
        z = _mm256_xor_si256(z, y);
        z = _mm256_xor_si256(z, x);
        return z;
    }

    /*
     * 256bit 用の出力変換
     */
    static inline void output_conv256(uint64_t array[4]) {
        __m256i x = load256(array);
        __m256i y = _mm256_srli_epi64(x, TEMPER_SH1);
        y = _mm256_and_si256(y, tm1_256.simd256);
        x = _mm256_xor_si256(x, y);
        y = _mm256_slli_epi64(x, TEMPER_SH2);
        y = _mm256_and_si256(y, tm2_256.simd256);
        x = _mm256_xor_si256(x, y);
        y = _mm256_slli_epi64(x, TEMPER_SH3);
        y = _mm256_and_si256(y, tm3_256.simd256);
        x = _mm256_xor_si256(x, y);
        y = _mm256_srli_epi64(x, TEMPER_SH4);
        x = _mm256_xor_si256(x, y);
        store256(array, x);
    }
#endif

#if HAVE_SSSE3
    /*
     * SSSE3 による 128 bit用のload
     */
    static inline __m128i load128(uint64_t * addr)
    {
        return _mm_load_si128((__m128i *)addr);
    }

    /*
     * SSSE3 による 128 bit用の整列されていないアドレスからのload
     * これは SSSE3 以後の機能
     */
    static inline __m128i loadu128(uint64_t * addr)
    {
        //return _mm_loadu_si128((__m128i *)addr);
        return _mm_lddqu_si128((__m128i *)addr);
    }

    /*
     * SSSE3 による 128 bit用のstore
     */
    static inline void store128(uint64_t * addr, __m128i x)
    {
        _mm_store_si128((__m128i *)addr, x);
    }

#if HAVE_SSE4_1
    /*
     * SSSE4.1 による128ビット漸化式関数
     */
    static inline __m128i recursion128sse41(__m128i x, __m128i y, __m128i z)
    {
        x = _mm_and_si128(x, um128.simd128);
        y = _mm_and_si128(y, lm128.simd128);
        x = _mm_or_si128(x, y);
        y = _mm_srli_epi64(x, 1);
        x = _mm_and_si128(x, ones128.simd128);
        // SSE4.1
        x = _mm_cmpeq_epi64(x, ones128.simd128);
        x = _mm_and_si128(x, mata128.simd128);
        z = _mm_xor_si128(z, y);
        z = _mm_xor_si128(z, x);
        return z;
    }
#endif

    /*
     * SSSE3 による128ビット漸化式関数
     */
    static inline __m128i recursion128ssse3(__m128i x, __m128i y, __m128i z)
    {
#define SHUFFLE_COPY_LOW 0xa0
        // calculate mata
        __m128i xx = _mm_and_si128(y, ones128.simd128);
        xx = _mm_cmpeq_epi32(xx, ones128.simd128);
        // 下位32ビットを上位32ビットにコピーする
        xx = _mm_shuffle_epi32(xx, SHUFFLE_COPY_LOW);
        xx = _mm_and_si128(xx, mata128.simd128);

        x = _mm_and_si128(x, um128.simd128);
        __m128i yy = _mm_and_si128(y, lm128.simd128);
        x = _mm_or_si128(x, yy);
        yy = _mm_srli_epi64(x, 1);

        z = _mm_xor_si128(z, yy);
        z = _mm_xor_si128(z, xx);
        return z;
#undef SHUFFLE_COPY_LOW
    }

    /**
     * 128 bit 出力変換
     */
    static inline void output_conv128(uint64_t * array) {
        __m128i x = load128(array);
        __m128i y = _mm_srli_epi64(x, TEMPER_SH1);
        y = _mm_and_si128(y, tm1_128.simd128);
        x = _mm_xor_si128(x, y);
        y = _mm_slli_epi64(x, TEMPER_SH2);
        y = _mm_and_si128(y, tm2_128.simd128);
        x = _mm_xor_si128(x, y);
        y = _mm_slli_epi64(x, TEMPER_SH3);
        y = _mm_and_si128(y, tm3_128.simd128);
        x = _mm_xor_si128(x, y);
        y = _mm_srli_epi64(x, TEMPER_SH4);
        x = _mm_xor_si128(x, y);
        store128(array, x);
    }
#endif // HAVE_SSSE3

    /*
     * SIMDを使わない64bit漸化式関数
     */
    static inline uint64_t recursion64(uint64_t x, uint64_t y, uint64_t z) {
        x = (x & UPPER_MASK) | (y & LOWER_MASK);
#if defined(NOT_USE_MAG_ARRAY)
        return z ^ (x >> 1)
            ^ static_cast<uint64_t>((-static_cast<int64_t>(x & 1)) & mata64);
#else
        return z ^ (x >> 1) ^ mag[x & 1];
#endif
    }

    /*
     * 出力変換
     */
    static inline void output_conv64(uint64_t array[1]) {
        uint64_t x = array[0];
        x ^= (x >> TEMPER_SH1) & TEMPER_MASK1;
        x ^= (x << TEMPER_SH2) & TEMPER_MASK2;
        x ^= (x << TEMPER_SH3) & TEMPER_MASK3;
        x ^= (x >> TEMPER_SH4);
        array[0] = x;
    }

/* =================
   状態空間まとめて生成
   ================= */

#if HAVE_AVX512F
    /*
     * 512 bit 用の状態空間まとめて生成
     */
    void fillState512avx512f(uint64_t * state)
    {
        int i = 0;
        __m512i x;
        __m512i y;
        __m512i z;
        __m512i w;
        x = load512(&state[i]);
        y = loadu512(&state[i + 1]);
        z = loadu512(&state[i + pos]);// pos は 8 で割れない
        w = recursion512(x, y, z);
        store512(&state[i], w);
        store512(&state[size], w);
        i = 8;
        for (; i + pos < size; i += 8) {
            x = load512(&state[i]);
            y = loadu512(&state[i + 1]);
            z = loadu512(&state[i + pos]);
            w = recursion512(x, y, z);
            store512(&state[i], w);
        }
        for (; i < size; i += 8) {
            x = load512(&state[i]);
            y = loadu512(&state[i + 1]);
            z = loadu512(&state[i + pos - size]);
            w = recursion512(x, y, z);
            store512(&state[i], w);
        }
    }
#else
    void fillState512avx512f(uint64_t *)
    {
        throw new std::logic_error("should not be called");
    }
#endif

#if HAVE_AVX2
    /*
     * 256 bit 用の状態空間まとめて生成
     */
    void fillState256avx2(uint64_t * state)
    {
        int i = 0;
        __m256i x;
        __m256i y;
        __m256i z;
        __m256i w;
        x = load256(&state[i]);
        y = loadu256(&state[i + 1]);
        z = load256(&state[i + pos]);
        w = recursion256(x, y, z);
        store256(&state[i], w);
        store256(&state[size], w);
        i = 4;
        for (; i + pos < size; i += 4) {
            x = load256(&state[i]);
            y = loadu256(&state[i + 1]);
            z = load256(&state[i + pos]);
            w = recursion256(x, y, z);
            store256(&state[i], w);
        }
        for (; i < size; i += 4) {
            x = load256(&state[i]);
            y = loadu256(&state[i + 1]);
            z = load256(&state[i + pos - size]);
            w = recursion256(x, y, z);
            store256(&state[i], w);
        }
        _mm256_zeroall();
    }
#else
    void fillState256avx2(uint64_t *)
    {
        throw new std::logic_error("should not be called");
    }
#endif

#if HAVE_SSE4_1
    /**
     * SSE4.2 用の状態空間まとめて生成
     */
    void fillState128sse41(uint64_t * state)
    {
        int i = 0;
        __m128i x;
        __m128i y;
        __m128i z;
        __m128i w;
        x = load128(&state[i]);
        y = loadu128(&state[i + 1]);
        z = load128(&state[i + pos]);
        w = recursion128sse41(x, y, z);
        store128(&state[i], w);
        store128(&state[size], w);
        i = 2;
        for (; i + pos < size; i += 2) {
            x = load128(&state[i]);
            y = loadu128(&state[i + 1]);
            z = load128(&state[i + pos]);
            w = recursion128sse41(x, y, z);
            store128(&state[i], w);
        }
        for (; i < size; i += 2) {
            x = load128(&state[i]);
            y = loadu128(&state[i + 1]);
            z = load128(&state[i + pos - size]);
            w = recursion128sse41(x, y, z);
            store128(&state[i], w);
        }
    }
#else
    void fillState128sse41(uint64_t *)
    {
        throw new std::logic_error("should not be called");
    }
#endif

#if HAVE_SSSE3
    /*
     * SSSE3 用の状態空間まとめて生成
     */
    void fillState128ssse3(uint64_t * state)
    {
        int i = 0;
        __m128i x;
        __m128i y;
        __m128i z;
        __m128i w;
        x = load128(&state[i]);
        y = loadu128(&state[i + 1]);
        z = load128(&state[i + pos]);
        w = recursion128ssse3(x, y, z);
        store128(&state[i], w);
        store128(&state[size], w);
        i = 2;
        for (; i + pos < size; i += 2) {
            x = load128(&state[i]);
            y = loadu128(&state[i + 1]);
            z = load128(&state[i + pos]);
            w = recursion128ssse3(x, y, z);
            store128(&state[i], w);
        }
        for (; i < size; i += 2) {
            x = load128(&state[i]);
            y = loadu128(&state[i + 1]);
            z = load128(&state[i + pos - size]);
            w = recursion128ssse3(x, y, z);
            store128(&state[i], w);
        }
    }
#else
    void fillState128ssse3(uint64_t *)
    {
        throw new std::logic_error("should not be called");
    }
#endif

    /*
     * SIMD を使わない状態空間まとめて生成
     */
    void fillState64(uint64_t * state)
    {
        int i = 0;

        for (; i + pos < size; i++) {
            state[i] = recursion64(state[i], state[i + 1], state[i + pos]);
        }
        for (; i < size - 1; i++) {
            state[i] = recursion64(state[i], state[i + 1],
                                   state[i + pos - size]);
        }
        state[size - 1] = recursion64(state[size - 1], state[0],
                                      state[pos - 1]);
    }

/* =========
   配列に生成
   ========= */
#if HAVE_AVX512F
    /*
     * AVX512F を使った512ビット用の配列に生成
     */
    void fillArray512avx512f(uint64_t * state, uint64_t array[], int length)
    {
        int i = 0;
        __m512i x;
        __m512i y;
        __m512i z;
        __m512i w;
        x = load512(&state[i]);
        y = loadu512(&state[i + 1]);
        z = loadu512(&state[i + pos]); // 512 だけ loadu
        w = recursion512(x, y, z);
        store512(&array[i], w);
        store512(&state[size], w);
        i = 8;
        for (; i + pos < size; i += 8) {
            x = load512(&state[i]);
            y = loadu512(&state[i + 1]);
            z = loadu512(&state[i + pos]);
            w = recursion512(x, y, z);
            store512(&array[i], w);
        }
        for (; i < size; i += 8) {
            x = load512(&state[i]);
            y = loadu512(&state[i + 1]);
            z = loadu512(&array[i + pos - size]);
            w = recursion512(x, y, z);
            store512(&array[i], w);
        }
        for (; i < length - size; i += 8) {
            x = load512(&array[i - size]);
            y = loadu512(&array[i + 1 - size]);
            z = loadu512(&array[i + pos - size]);
            w = recursion512(x, y, z);
            store512(&array[i], w);
            output_conv512(&array[i - size]);
        }
        for (; i < length; i += 8) {
            x = load512(&array[i - size]);
            y = loadu512(&array[i + 1 - size]);
            z = loadu512(&array[i + pos - size]);
            w = recursion512(x, y, z);
            store512(&array[i], w);
            output_conv512(&array[i - size]);
        }
        int j = 0;
        for (i = i - size; i < length; i += 8) {
            w = load512(&array[i]);
            store512(&state[j], w);
            output_conv512(&array[i]);
            j += 8;
        }
    }
#else
    void fillArray512avx512f(uint64_t *, uint64_t *, int )
    {
        throw new std::logic_error("should not be called");
    }
#endif

#if HAVE_AVX2
    /*
     * AVX2 を使った256ビット用の配列に生成
     */
    void fillArray256avx2(uint64_t * state, uint64_t array[], int length)
    {
        int i = 0;
        __m256i x;
        __m256i y;
        __m256i z;
        __m256i w;
        x = load256(&state[i]);
        y = loadu256(&state[i + 1]);
        z = load256(&state[i + pos]);
        w = recursion256(x, y, z);
        store256(&array[i], w);
        store256(&state[size], w);
        i = 4;
        for (; i + pos < size; i += 4) {
            x = load256(&state[i]);
            y = loadu256(&state[i + 1]);
            z = load256(&state[i + pos]);
            w = recursion256(x, y, z);
            store256(&array[i], w);
        }
        for (; i < size; i += 4) {
            x = load256(&state[i]);
            y = loadu256(&state[i + 1]);
            z = load256(&array[i + pos - size]);
            w = recursion256(x, y, z);
            store256(&array[i], w);
        }
        for (; i < length - size; i += 4) {
            x = load256(&array[i - size]);
            y = loadu256(&array[i + 1 - size]);
            z = load256(&array[i + pos - size]);
            w = recursion256(x, y, z);
            store256(&array[i], w);
            output_conv256(&array[i - size]);
        }
        for (; i < length; i += 4) {
            x = load256(&array[i - size]);
            y = loadu256(&array[i + 1 - size]);
            z = load256(&array[i + pos - size]);
            w = recursion256(x, y, z);
            store256(&array[i], w);
            output_conv256(&array[i - size]);
        }
        int j = 0;
        for (i = i - size; i < length; i += 4) {
            w = load256(&array[i]);
            store256(&state[j], w);
            output_conv256(&array[i]);
            j += 4;
        }
        _mm256_zeroall();
    }
#else
    void fillArray256avx2(uint64_t *, uint64_t *, int )
    {
        throw new std::logic_error("should not be called");
    }
#endif

#if HAVE_SSE4_1
    void fillArray128sse41(uint64_t * state, uint64_t array[], int length)
    {
        int i = 0;
        __m128i x;
        __m128i y;
        __m128i z;
        __m128i w;
        x = load128(&state[i]);
        y = loadu128(&state[i + 1]);
        z = load128(&state[i + pos]);
        w = recursion128sse41(x, y, z);
        store128(&array[i], w);
        store128(&state[size], w);
        i = 2;
        for (; i + pos < size; i += 2) {
            x = load128(&state[i]);
            y = loadu128(&state[i + 1]);
            z = load128(&state[i + pos]);
            w = recursion128sse41(x, y, z);
            store128(&array[i], w);
        }
        for (; i < size; i += 2) {
            x = load128(&state[i]);
            y = loadu128(&state[i + 1]);
            z = load128(&array[i + pos - size]);
            w = recursion128sse41(x, y, z);
            store128(&array[i], w);
        }
        for (; i < length - size; i += 2) {
            x = load128(&array[i - size]);
            y = loadu128(&array[i + 1 - size]);
            z = load128(&array[i + pos - size]);
            w = recursion128sse41(x, y, z);
            store128(&array[i], w);
            output_conv128(&array[i - size]);
        }
        for (; i < length; i += 2) {
            x = load128(&array[i - size]);
            y = loadu128(&array[i + 1 - size]);
            z = load128(&array[i + pos - size]);
            w = recursion128sse41(x, y, z);
            store128(&array[i], w);
            output_conv128(&array[i - size]);
        }
        int j = 0;
        for (i = length - size; i < length; i += 2) {
            w = load128(&array[i]);
            store128(&state[j], w);
            output_conv128(&array[i]);
            j += 2;
        }
    }
#else
    void fillArray128sse41(uint64_t *, uint64_t *, int)
    {
        throw new std::logic_error("should not be called");
    }
#endif

#if HAVE_SSSE3
    void fillArray128ssse3(uint64_t * state, uint64_t array[], int length)
    {
        int i = 0;
        __m128i x;
        __m128i y;
        __m128i z;
        __m128i w;
        x = load128(&state[i]);
        y = loadu128(&state[i + 1]);
        z = load128(&state[i + pos]);
        w = recursion128ssse3(x, y, z);
        store128(&array[i], w);
        store128(&state[size], w);
        i = 2;
        for (; i + pos < size; i += 2) {
            x = load128(&state[i]);
            y = loadu128(&state[i + 1]);
            z = load128(&state[i + pos]);
            w = recursion128ssse3(x, y, z);
            store128(&array[i], w);
        }
        for (; i < size; i += 2) {
            x = load128(&state[i]);
            y = loadu128(&state[i + 1]);
            z = load128(&array[i + pos - size]);
            w = recursion128ssse3(x, y, z);
            store128(&array[i], w);
        }
        for (; i < length - size; i += 2) {
            x = load128(&array[i - size]);
            y = loadu128(&array[i + 1 - size]);
            z = load128(&array[i + pos - size]);
            w = recursion128ssse3(x, y, z);
            store128(&array[i], w);
            output_conv128(&array[i - size]);
        }
        for (; i < length; i += 2) {
            x = load128(&array[i - size]);
            y = loadu128(&array[i + 1 - size]);
            z = load128(&array[i + pos - size]);
            w = recursion128ssse3(x, y, z);
            store128(&array[i], w);
            output_conv128(&array[i - size]);
        }
        int j = 0;
        for (i = length - size; i < length; i += 2) {
            w = load128(&array[i]);
            store128(&state[j], w);
            output_conv128(&array[i]);
            j += 2;
        }
    }
#else
    void fillArray128ssse3(uint64_t *, uint64_t *, int)
    {
        throw new std::logic_error("should not be called");
    }
#endif

    void fillArray64(uint64_t * state, uint64_t array[], int length)
    {
        int i = 0;

        for (; i + pos < size; i++) {
            array[i] = recursion64(state[i], state[i + 1], state[i + pos]);
        }
        for (; i < size - 1; i++) {
            array[i] = recursion64(state[i], state[i + 1],
                                   array[i + pos - size]);
        }
        // i = 311
        array[i] = recursion64(state[size - 1], array[0],
                               array[pos - 1]);
        i++;
        for (; i < length - size; i++) {
            array[i] = recursion64(array[i - size], array[i + 1 - size],
                                   array[i + pos - size]);
            output_conv64(&array[i - size]);
        }
        for (; i < length; i++) {
            array[i] = recursion64(array[i - size], array[i + 1 - size],
                                   array[i + pos - size]);
            output_conv64(&array[i - size]);
        }
        int j = 0;
        for (i = length - size; i < length; i++) {
            state[j++] = array[i];
            output_conv64(&array[i]);
        }
    }

    uint64_t original_generation(uint64_t st[], int *index)
    {
        static const uint64_t mag01[2]={UINT64_C(0), MATA};
        *index = *index % size;
        uint64_t x = (st[*index] & UPPER_MASK)
            | (st[(*index + 1) % size] & LOWER_MASK);
        st[*index] = st[(*index + pos) % size] ^ (x >> 1)
            ^ mag01[(int)(x & UINT64_C(1))];
        x = st[*index];
        (*index)++;
        x ^= (x >> 29) & UINT64_C(0x5555555555555555);
        x ^= (x << 17) & UINT64_C(0x71D67FFFEDA60000);
        x ^= (x << 37) & UINT64_C(0xFFF7EEE000000000);
        x ^= (x >> 43);
        return x;
    }

    void original_init(uint64_t mt[], uint64_t seed)
    {
        int mti;
        mt[0] = seed;
        for (mti=1; mti < size; mti++) {
            mt[mti] = (UINT64_C(6364136223846793005)
                       * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
        }
    }

    void original_init_by_array(uint64_t mt[], uint64_t init_key[],
                                uint64_t key_length)
    {
        uint64_t NN = static_cast<uint64_t>(size);
        unsigned int i, j;
        uint64_t k;
        original_init(mt, UINT64_C(19650218));
        i=1; j=0;
        k = (NN>key_length ? NN : key_length);
        for (; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62))
                              * UINT64_C(3935559000370003845)))
                + init_key[j] + (uint64_t)j; /* non linear */
            i++; j++;
            if (i>=NN) { mt[0] = mt[NN-1]; i=1; }
            if (j>=key_length) j=0;
        }
        for (k=NN-1; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62))
                              * UINT64_C(2862933555777941757)))
                - (uint64_t)i; /* non linear */
            i++;
            if (i>=NN) { mt[0] = mt[NN-1]; i=1; }
        }
        /* MSB is 1; assuring non-zero initial array */
        mt[0] = UINT64_C(1) << 63;
    }
}

/* ==========================================
   MersenneTwisterMultiple64 Member functions
   ========================================== */
namespace MersenneTwister {
    using namespace std;

    bool MT19937_64::selfTest()
    {
        uint64_t init[4] = {UINT64_C(0x12345), UINT64_C(0x23456),
                            UINT64_C(0x34567), UINT64_C(0x45678)};
        int length = 4;
        uint64_t test_state[size];
        MT19937_64 mt(0);
        original_init(test_state, 0);
        for (int i = 0; i < size; i++) {
            if (mt.state[i] != test_state[i]) {
                printf("1 state mismatch i = %d mt.state[i] = %" PRIx64
                       " test_state[i] = %" PRIx64"\n",
                       i, mt.state[i], test_state[i]);
                fflush(stdout);
                return false;
            }
        }
        int index = 0;
        for (int i = 0; i < 1000; i++) {
            uint64_t x = mt.generate();
            uint64_t y = original_generation(test_state, &index);
            if (x != y) {
                printf("1 generate mismatch i = %d,\n", i);
                printf("mt = %" PRIu64 "(%" PRIx64 ")\n", x, x);
                printf("orig = %" PRIu64 "(%" PRIx64 ")\n", y, y);
                fflush(stdout);
                return false;
            }
        }

        original_init_by_array(test_state, init, length);
        mt.seed(init, length);
        for (int i = 0; i < size; i++) {
            if (mt.state[i] != test_state[i]) {
                printf("state mismatch i = %d mt.state[i] = %" PRIx64
                       " test_state[i] = %" PRIx64"\n",
                       i, mt.state[i], test_state[i]);
                fflush(stdout);
                return false;
            }
        }
        index = 0;
        for (int i = 0; i < 1000; i++) {
            uint64_t x = mt.generate();
            uint64_t y = original_generation(test_state, &index);
            if (x != y) {
                printf("generate mismatch i = %d,\n", i);
                printf("mt = %" PRIu64 "(%" PRIx64 ")\n", x, x);
                printf("orig = %" PRIu64 "(%" PRIx64 ")\n", y, y);
                fflush(stdout);
                return false;
            }
        }
        return true;
    }

    MT19937_64::MT19937_64(uint64_t seedValue) {
        cf = cpu_feature();
        int add = 0;
        if (cf.avx512f) {
            DMSG("AVX512F");
            add = 8;
        } else if (cf.avx2) {
            DMSG("AVX2");
            add = 4;
        } else if (cf.sse4_2) {
            DMSG("SSE4_2");
            add = 2;
        } else if (cf.ssse3) {
            DMSG("SSSE3");
            add = 2;
        } else {
            DMSG("DEFAULT64");
            add = 0;
        }
        size_t memsize = (size + add) * sizeof(uint64_t);
        state = alignedAlloc<uint64_t *>(memsize);
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue);
    }

    MT19937_64::~MT19937_64() {
        alignedFree(state);
    }

    void MT19937_64::seed(uint64_t seed)
    {
        state[0]= seed;
        for (int i = 1; i < size; i++) {
            state[i] = i
                + UINT64_C(6364136223846793005)
                * (state[i - 1] ^ (state[i - 1] >> 62));
        }
        index = size;
    }

    void MT19937_64::seed(uint64_t seedValue[], int length)
    {
        original_init_by_array(state, seedValue, length);
        index = size;
    }

    void MT19937_64::fillState()
    {
        if (cf.avx512f) {
            fillState512avx512f(state);
        } else if (cf.avx2) {
            fillState256avx2(state);
        } else if (cf.sse4_1) {
            fillState128sse41(state);
        } else if (cf.ssse3) {
            fillState128ssse3(state);
        } else {
            fillState64(state);
        }
        index = 0;
    }

    void MT19937_64::fillArray(uint64_t * array, int length)
    {
        if (length < size || index != size) {
            DMSG("fillArraySequential");
            for (int i = 0; i < length; i++) {
                array[i] = generate();
            }
            return;
        }
        int align = reinterpret_cast<uintptr_t>(array) % 256;
        if ((align % 64 == 0) && (length % 8 == 0) && cf.avx512f) {
            DMSG("fillArray512avx512f");
            fillArray512avx512f(state, array, length);
        } else if ((align % 32 == 0) && (length % 4 == 0) && cf.avx2) {
            DMSG("fillArray256avx2");
            fillArray256avx2(state, array, length);
        } else if ((align % 16 == 0) && (length % 2 == 0) && cf.sse4_1) {
            DMSG("fillArray128sse41");
            fillArray128sse41(state, array, length);
        } else if (align % 16 == 0 && (length % 2 == 0) && cf.ssse3) {
            DMSG("fillArray128ssse3");
            fillArray128ssse3(state, array, length);
        } else {
            DMSG("fillArray64");
            fillArray64(state, array, length);
        }
        index = size;
    }
}
