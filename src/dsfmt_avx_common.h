#pragma once
#ifndef SFMT_DIST_DSFMT_AVX_COMMON_H
#define SFMT_DIST_DSFMT_AVX_COMMON_H

#include "sfmt-dist.h"
#include "w256.h"

#include <sfmt-dist/cpu_feature.h>
#include <sfmt-dist/aligned_alloc.h>

namespace {
    const string getIDString()
    {
        stringstream ss;
        char delim = ':';
        ss << "dSFMTAVX-" << dec << DSFMTAVX_MEXP
           << ":" << dec << DSFMTAVX_POS1 << "-" << DSFMTAVX_SL1;
        for (int i = 0; i < 4; i++) {
            ss << delim << setfill('0') << setw(13) << hex
               << mask1.u64[i];
            delim = '-';
        }
        string s;
        ss >> s;
        return s;
    }

        /**
     * This function represents a function used in the initialization
     * by init_by_array
     * @param x 32-bit integer
     * @return 32-bit integer
     */
    uint32_t ini_func1(uint32_t x)
    {
        return (x ^ (x >> 27)) * UINT32_C(1664525);
    }

    /**
     * This function represents a function used in the initialization
     * by init_by_array
     * @param x 32-bit integer
     * @return 32-bit integer
     */
    uint32_t ini_func2(uint32_t x)
    {
        return (x ^ (x >> 27)) * UINT32_C(1566083941);
    }

    /**
     * This function certificate the period of 2^{SFMT_MEXP}-1.
     * @param dsfmt dsfmt state vector.
     */
    void period_certification(w256_t * state)
    {
        w256_t tmp;
        uint64_t inner;
        int size = DSFMTAVX_SIZE;
        for (int i = 0; i < 4; i++) {
            tmp.u64[i] = state[size].u64[i] ^ fix1.u64[i];
        }
        inner = 0;
        for (int i = 0; i < 4; i++) {
            inner ^= tmp.u64[i] & pcv1.u64[0];
        }
        inner ^= inner >> 32;
        inner ^= inner >> 16;
        inner ^= inner >> 8;
        inner ^= inner >> 4;
        inner ^= inner >> 2;
        inner ^= inner >> 1;
        inner &= 1;
        /* check OK */
        if (inner == 1) {
            return;
        }
        /* check NG, and modification */
        if ((pcv1.u64[0] & 1) == 1) {
            state[size].u64[0] ^= 1;
            return;
        }
        int i;
        int j;
        uint64_t work;
        for (i = 0; i < 4; i++) {
            work = 1;
            for (j = 0; j < 64; j++) {
                if ((work & pcv1.u64[i]) != 0) {
                    state[size].u64[i] ^= work;
                    return;
                }
                work = work << 1;
            }
        }
    }

    /**
     * This function initializes the internal state array to fit the IEEE
     * 754 format.
     * @param dsfmt dsfmt state vector.
     */
    void initial_mask(w256_t * state)
    {
        int i;
        uint64_t *psfmt;

        int size = DSFMTAVX_SIZE;
        psfmt = &state[0].u64[0];
        for (i = 0; i < size * 4; i++) {
            psfmt[i] = (psfmt[i] & DSFMTAVX_LOW_MASK) | DSFMTAVX_HIGH_CONST;
        }
    }

    /**
     * This function initializes the internal state array with a 32-bit
     * integer seed.
     * @param dsfmt dsfmt state vector.
     * @param seed a 32-bit integer used as the seed.
     * @param mexp caller's mersenne expornent
     */
    void init(w256_t * state, uint32_t seed) {
        int i;
        uint32_t *psfmt;

        int size = DSFMTAVX_SIZE;
        psfmt = &state[0].u32[0];
        psfmt[0] = seed;
        for (i = 1; i < (size + 1) * 8; i++) {
            psfmt[i] = 1812433253UL
                * (psfmt[i - 1] ^ (psfmt[i - 1] >> 30)) + i;
        }
        initial_mask(state);
        period_certification(state);
    }

    /**
     * This function initializes the internal state array,
     * with an array of 32-bit integers used as the seeds
     * @param dsfmt dsfmt state vector.
     * @param init_key the array of 32-bit integers, used as a seed.
     * @param key_length the length of init_key.
     * @param mexp caller's mersenne expornent
     */
    void init(w256_t * state, uint32_t init_key[], int key_length)
    {
        int i, j, count;
        uint32_t r;
        uint32_t *psfmt32;
        int lag;
        int mid;
        int size = (DSFMTAVX_SIZE + 1) * 2 * 4;   /* pulmonary */

        if (size >= 623) {
            lag = 11;
        } else if (size >= 68) {
            lag = 7;
        } else if (size >= 39) {
            lag = 5;
        } else {
            lag = 3;
        }
        mid = (size - lag) / 2;

        psfmt32 = &state[0].u32[0];
        // check here
        memset(state, 0x8b, sizeof(uint32_t) * size);
        if (key_length + 1 > size) {
            count = key_length + 1;
        } else {
            count = size;
        }
        r = ini_func1(psfmt32[0] ^ psfmt32[mid % size]
                      ^ psfmt32[(size - 1) % size]);
        psfmt32[mid % size] += r;
        r += key_length;
        psfmt32[(mid + lag) % size] += r;
        psfmt32[0] = r;
        count--;
        for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
            r = ini_func1(psfmt32[i]
                          ^ psfmt32[(i + mid) % size]
                          ^ psfmt32[(i + size - 1) % size]);
            psfmt32[(i + mid) % size] += r;
            r += init_key[j] + i;
            psfmt32[(i + mid + lag) % size] += r;
            psfmt32[i] = r;
            i = (i + 1) % size;
        }
        for (; j < count; j++) {
            r = ini_func1(psfmt32[i]
                          ^ psfmt32[(i + mid) % size]
                          ^ psfmt32[(i + size - 1) % size]);
            psfmt32[(i + mid) % size] += r;
            r += i;
            psfmt32[(i + mid + lag) % size] += r;
            psfmt32[i] = r;
            i = (i + 1) % size;
        }
        for (j = 0; j < size; j++) {
            r = ini_func2(psfmt32[i]
                          + psfmt32[(i + mid) % size]
                          + psfmt32[(i + size - 1) % size]);
            psfmt32[(i + mid) % size] ^= r;
            r -= i;
            psfmt32[(i + mid + lag) % size] ^= r;
            psfmt32[i] = r;
            i = (i + 1) % size;
        }
        initial_mask(state);
        period_certification(state);
    }

#if HAVE_AVX2
    /**
     * This function represents the recursion formula.
     * @param params parameters
     * @param a a 128-bit part of the interal state array
     * @param b a 128-bit part of the interal state array
     * @param c a 128-bit part of the interal state array
     * @param d a 128-bit part of the interal state array
     * @return result
     */
    inline __m256i recursion256(const __m256i mask,
                                __m256i a, __m256i b, __m256i * u)
    {
//        extern const w256x32_t perm256;
        __m256i x = _mm256_slli_epi64(a, DSFMTAVX_SL1);
        __m256i y = _mm256_permutevar8x32_epi32(*u, perm256.si256);
        __m256i z = _mm256_xor_si256(x, b);
        y = _mm256_xor_si256(y, z);
        __m256i v = _mm256_srli_epi64(y, DSFMTAVX_SR);
        __m256i w = _mm256_and_si256(y, mask);
        __m256i s = _mm256_xor_si256(a, v);
        *u = y;
        return _mm256_xor_si256(w, s);
    }

    inline  __m256d convert256_c0o1(__m256d w) {
        extern const w256xd_t m_one256;
        return _mm256_add_pd(w, m_one256.sd256);
    }

    inline  __m256d convert256_o0c1(__m256d w) {
        extern const w256xd_t two256;
        return _mm256_sub_pd(two256.sd256, w);
    }

    inline  __m256d convert256_o0o1(__m256i w) {
        extern const w256xd_t m_one256;
        extern const MersenneTwister::w256_t one256;
        w = _mm256_or_si256(w, one256.si256);
        return _mm256_add_pd((__m256d)w, m_one256.sd256);
    }
#endif

    /**
     * This function represents the recursion formula.
     * @param p parameter
     * @param r output
     * @param a a 256-bit part of the internal state array
     * @param b a 256-bit part of the internal state array
     * @param lung a 256-bit part of the internal state array
     */
    inline void recursion64(MersenneTwister::w256_t *r,
                            MersenneTwister::w256_t *a,
                            MersenneTwister::w256_t * b,
                            MersenneTwister::w256_t *lung)
    {
//        extern MersenneTwister::w256_t mask1;
//        extern w256_t mask1;
        uint32_t tmp;
        tmp = lung->u32[7];
        lung->u32[7] = lung->u32[6];
        lung->u32[6] = lung->u32[5];
        lung->u32[5] = lung->u32[4];
        lung->u32[4] = lung->u32[3];
        lung->u32[3] = lung->u32[2];
        lung->u32[2] = lung->u32[1];
        lung->u32[1] = lung->u32[0];
        lung->u32[0] = tmp;
        MersenneTwister::w256_t t;
        lung->u64[0] ^= b->u64[0];
        lung->u64[1] ^= b->u64[1];
        lung->u64[2] ^= b->u64[2];
        lung->u64[3] ^= b->u64[3];
        lung->u64[0] ^= a->u64[0] << DSFMTAVX_SL1;
        lung->u64[1] ^= a->u64[1] << DSFMTAVX_SL1;
        lung->u64[2] ^= a->u64[2] << DSFMTAVX_SL1;
        lung->u64[3] ^= a->u64[3] << DSFMTAVX_SL1;
        t.u64[0] = lung->u64[0] >> DSFMTAVX_SR;
        t.u64[1] = lung->u64[1] >> DSFMTAVX_SR;
        t.u64[2] = lung->u64[2] >> DSFMTAVX_SR;
        t.u64[3] = lung->u64[3] >> DSFMTAVX_SR;
        t.u64[0] ^= a->u64[0];
        t.u64[1] ^= a->u64[1];
        t.u64[2] ^= a->u64[2];
        t.u64[3] ^= a->u64[3];
        r->u64[0] = t.u64[0] ^ (lung->u64[0] & mask1.u64[0]);
        r->u64[1] = t.u64[1] ^ (lung->u64[1] & mask1.u64[1]);
        r->u64[2] = t.u64[2] ^ (lung->u64[2] & mask1.u64[2]);
        r->u64[3] = t.u64[3] ^ (lung->u64[3] & mask1.u64[3]);
    }

    inline  void convert64_c0o1(MersenneTwister::w256_t *w) {
        w->d[0] -= 1.0;
        w->d[1] -= 1.0;
        w->d[2] -= 1.0;
        w->d[3] -= 1.0;
    }

    inline  void convert64_o0c1(MersenneTwister::w256_t *w) {
        w->d[0] = 2.0 - w->d[0];
        w->d[1] = 2.0 - w->d[1];
        w->d[2] = 2.0 - w->d[2];
        w->d[3] = 2.0 - w->d[3];
    }

    inline  void convert64_o0o1(MersenneTwister::w256_t *w)
    {
        w->u64[0] |= 1;
        w->u64[1] |= 1;
        w->u64[2] |= 1;
        w->u64[3] |= 1;
        w->d[0] -= 1.0;
        w->d[1] -= 1.0;
        w->d[2] -= 1.0;
        w->d[3] -= 1.0;
    }
//}
#if HAVE_AVX2
    /**
     * This function fills the internal state array with pseudorandom
     * integers.
     * @param p parameter
     * @param state SFMT internal state
     */
    void fillState256(double * state) {
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        __m256i lung = pstate[DSFMTAVX_SIZE].si256;
        __m256i mask = mask1.si256;
        int i = 0;
        for (;i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            pstate[i].si256 = recursion256(mask,
                                           pstate[i].si256,
                                           pstate[i + DSFMTAVX_POS1].si256,
                                           &lung);
        }
        for (;i < DSFMTAVX_SIZE; i++) {
            pstate[i].si256
                = recursion256(mask,
                               pstate[i].si256,
                               pstate[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                               &lung);
        }
        pstate[DSFMTAVX_SIZE].si256 = lung;
        _mm256_zeroall();
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     *
     * @param p parameter
     * @param state SFMT internal state.
     * @param array64 an 256-bit array to be filled by pseudorandom numbers.
     * @param length number of 256-bit pseudorandom numbers to be generated.
     */
    void fillArray256_c1o2(double * state,
                           double * array64, int length)
    {
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        const __m256i mask = mask1.si256;
        __m256i lung = pstate[2].si256;
//            __m256i a = pstate[0].si256;
//            __m256i b = pstate[1].si256;
        int i = 0;
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            array[i].si256 = recursion256(mask,
                                          pstate[i].si256,
                                          pstate[i + DSFMTAVX_POS1].si256,
                                          &lung);
        }
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            array[i].si256
                = recursion256(mask,
                               pstate[i].si256,
                               array[i + DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                               &lung);
        }
        for (; i < length; i++) {
            array[i].si256
                = recursion256(mask,
                               array[i - DSFMTAVX_SIZE].si256,
                               array[i + DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                               &lung);
        }
        int j = 0;
        for (int i = length - DSFMTAVX_SIZE; i < length ;i++) {
            pstate[j++].si256 = array[i].si256;
        }
        pstate[DSFMTAVX_SIZE].si256 = lung;
        _mm256_zeroall();
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param p parameter
     * @param state SFMT internal state.
     * @param array64 an 256-bit array to be filled by pseudorandom numbers.
     * @param length number of 256-bit pseudorandom numbers to be generated.
     */
    void fillArray256_c0o1(double * state, double * array64,
                           int length)
    {
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        const __m256i mask = mask1.si256;
        __m256i lung = pstate[2].si256;
//            __m256i a = pstate[0].si256;
//            __m256i b = pstate[1].si256;
        int i = 0;
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            array[i].si256 = recursion256(mask,
                                          pstate[i].si256,
                                          pstate[i + DSFMTAVX_POS1].si256,
                                          &lung);
        }
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            array[i].si256
                = recursion256(mask,
                               pstate[i].si256,
                               array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                               &lung);
        }
        for (; i < length; i++) {
            array[i].si256
                = recursion256(mask,
                               array[i - DSFMTAVX_SIZE].si256,
                               array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                               &lung);
            convert256_c0o1(array[i - DSFMTAVX_SIZE].sd256);
        }
        int j = 0;
        for (int i = length - DSFMTAVX_SIZE; i < length ;i++) {
            pstate[j++].si256 = array[i].si256;
            convert256_c0o1(array[i].sd256);
        }
        pstate[DSFMTAVX_SIZE].si256 = lung;
        _mm256_zeroall();
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param p parameter
     * @param state SFMT internal state.
     * @param array64 an 256-bit array to be filled by pseudorandom numbers.
     * @param length number of 256-bit pseudorandom numbers to be generated.
     */
    void fillArray256_o0c1(double * state, double * array64,
                           int length)
    {
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        const __m256i mask = mask1.si256;
        __m256i lung = pstate[2].si256;
//            __m256i a = pstate[0].si256;
//            __m256i b = pstate[1].si256;
        int i = 0;
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            array[i].si256 = recursion256(mask,
                                          pstate[i].si256,
                                          pstate[i + DSFMTAVX_POS1].si256,
                                          &lung);
        }
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            array[i].si256
                = recursion256(mask,
                               pstate[i].si256,
                               array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                               &lung);
        }
        for (; i < length; i++) {
            array[i].si256
                = recursion256(mask,
                               array[i - DSFMTAVX_SIZE].si256,
                               array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                               &lung);
            convert256_o0c1(array[i - DSFMTAVX_SIZE].sd256);
        }
        int j = 0;
        for (int i = length - DSFMTAVX_SIZE; i < length ;i++) {
            pstate[j++].si256 = array[i].si256;
            convert256_o0c1(array[i].sd256);
        }
        pstate[DSFMTAVX_SIZE].si256 = lung;
        _mm256_zeroall();
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param p parameter
     * @param state SFMT internal state.
     * @param array64 an 256-bit array to be filled by pseudorandom numbers.
     * @param length number of 256-bit pseudorandom numbers to be generated.
     */
    void fillArray256_o0o1(double * state, double * array64,
                           int length)
    {
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        const __m256i mask = mask1.si256;
        __m256i lung = pstate[2].si256;
//            __m256i a = pstate[0].si256;
//            __m256i b = pstate[1].si256;
        int i = 0;
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            array[i].si256 = recursion256(mask,
                                          pstate[i].si256,
                                          pstate[i + DSFMTAVX_POS1].si256,
                                          &lung);
        }
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            array[i].si256
                = recursion256(mask,
                               pstate[i].si256,
                               array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                               &lung);
        }
        for (; i < length; i++) {
            array[i].si256
                = recursion256(mask,
                               array[i - DSFMTAVX_SIZE].si256,
                               array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                               &lung);
            convert256_o0o1(array[i - DSFMTAVX_SIZE].si256);
        }
        int j = 0;
        for (int i = length - DSFMTAVX_SIZE; i < length ;i++) {
            pstate[j++].si256 = array[i].si256;
            convert256_o0o1(array[i].si256);
        }
        pstate[DSFMTAVX_SIZE].si256 = lung;
        _mm256_zeroall();
    }

#else // don't HAVE_AVX
    void fillState256(double *)
    {
        throw new std::logic_error("should not be called");
    }

    void fillArray256_c1o2(double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray256_c0o1(double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray256_o0c1(double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray256_o0o1(double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
#endif // HAVE_AVX2

    /**
     * This function fills the internal state array with pseudorandom
     * integers.
     * @param p parameter
     * @param sfmt SFMT internal state
     */
    void fillState64(double * state64)
    {
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t lung = state[DSFMTAVX_SIZE];
        int i = 0;
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            recursion64(&state[i], &state[i],
                        &state[i + DSFMTAVX_POS1], &lung);
        }
        for (; i < DSFMTAVX_SIZE; i++) {
            recursion64(&state[i], &state[i],
                        &state[i + DSFMTAVX_POS1 - DSFMTAVX_SIZE], &lung);
        }
        state[DSFMTAVX_SIZE] = lung;
    }

    void fillArray64_c1o2(double * state64,
                          double * array64, int length)
    {
        DMSG("start fillArray64_c1o2");
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t *lung = &state[2];
        int i = 0;
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            recursion64(&array[i], &state[i],
                        &state[i + DSFMTAVX_POS1], lung);
        }
        for (; i < DSFMTAVX_SIZE; i++) {
            recursion64(&array[i], &state[i],
                        &array[i + DSFMTAVX_POS1 - DSFMTAVX_SIZE], lung);
        }
        for (; i < length; i++) {
            recursion64(&array[i], &array[i - DSFMTAVX_SIZE],
                        &array[i + DSFMTAVX_POS1 - DSFMTAVX_SIZE], lung);
        }
        int j = 0;
        for (i = length - DSFMTAVX_SIZE; i < length; i++) {
            state[j++] = array[i];
        }
        state[DSFMTAVX_SIZE] = *lung;
    }

    void fillArray64_c0o1(double * state64,
                          double * array64, int length)
    {
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t *lung = &state[2];
        int i = 0;
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            recursion64(&array[i], &state[i],
                        &state[i + DSFMTAVX_POS1], lung);
        }
        for (; i < DSFMTAVX_SIZE; i++) {
            recursion64(&array[i], &state[i],
                        &array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE], lung);
        }
        for (; i < length; i++) {
            recursion64(&array[i], &array[i - DSFMTAVX_SIZE],
                        &array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE], lung);
            convert64_c0o1(&array[i - DSFMTAVX_SIZE]);
        }
        int j = 0;
        for (i = length - DSFMTAVX_SIZE; i < length; i++) {
            state[j++] = array[i];
            convert64_c0o1(&array[i]);
        }
        state[DSFMTAVX_SIZE] = *lung;
    }

    void fillArray64_o0c1(double * state64,
                          double * array64, int length)
    {
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t *lung = &state[2];
        int i = 0;
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            recursion64(&array[i], &state[i],
                        &state[i + DSFMTAVX_POS1], lung);
        }
        for (; i < DSFMTAVX_SIZE; i++) {
            recursion64(&array[i], &state[i],
                        &array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE], lung);
        }
        for (; i < length; i++) {
            recursion64(&array[i], &array[i - DSFMTAVX_SIZE],
                        &array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE], lung);
            convert64_o0c1(&array[i - DSFMTAVX_SIZE]);
        }
        int j = 0;
        for (i = length - DSFMTAVX_SIZE; i < length; i++) {
            state[j++] = array[i];
            convert64_o0c1(&array[i]);
        }
        state[DSFMTAVX_SIZE] = *lung;
    }

    void fillArray64_o0o1(double * state64,
                          double * array64, int length)
    {
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t *lung = &state[2];
        int i = 0;
        for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
            recursion64(&array[i], &state[i],
                        &state[i + DSFMTAVX_POS1], lung);
        }
        for (; i < DSFMTAVX_SIZE; i++) {
            recursion64(&array[i], &state[i],
                        &array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE], lung);
        }
        for (; i < length; i++) {
            recursion64(&array[i], &array[i - DSFMTAVX_SIZE],
                        &array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE], lung);
            convert64_o0o1(&array[i - DSFMTAVX_SIZE]);
        }
        int j = 0;
        for (i = length - DSFMTAVX_SIZE; i < length; i++) {
            state[j++] = array[i];
            convert64_o0o1(&array[i]);
        }
        state[DSFMTAVX_SIZE] = *lung;
    }

#if HAVE_AVX2
    inline __m128i do_uniform256(__m256d a, __m256d max256, __m128i min128)
    {
        a = _mm256_add_pd(a, m_one256.sd256);
        a = _mm256_mul_pd(a, max256);
        __m128i y = _mm256_cvttpd_epi32(a); // always truncate
        return _mm_add_epi32(y, min128);
    }

    void fillArray256_maxint(double * state, int32_t *array32,
                             uint64_t rmax, int32_t min)
    {
        DMSG("fillArray256_maxint step 1");
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        __m128i min128 = _mm_set1_epi32(min);
        __m256d max256 = _mm256_set1_pd(static_cast<double>(rmax));
        __m256d a;
        fillState256(state);
        DMSG("fillArray256_maxint step 2");
        int j = 0;
        for (int i = 0; i < DSFMTAVX_SIZE; i++) {
            a = pstate[i].sd256;
            array[j++].si128 = do_uniform256(a, max256, min128);
        }
        fillState256(state);
        for (int i = 0; i < DSFMTAVX_SIZE; i++) {
            a = pstate[i].sd256;
            array[j++].si128 = do_uniform256(a, max256, min128);
        }
        _mm256_zeroall();
    }

    int fillArray256_boxmuller(double * state, double * array,
                               double mu, double sigma)
    {
        w256_t * state256 = reinterpret_cast<w256_t *>(state);
        w256_t * array256 = reinterpret_cast<w256_t *>(array);
        w256_t w;
        __m128d axy[DSFMTAVX_SIZE * 2];
        double ar[DSFMTAVX_SIZE * 2];
        __m256d * paxy = reinterpret_cast<__m256d *>(axy);
        fillState256(state);
        int p = 0;
        for (int i = 0; i < DSFMTAVX_SIZE; i++) {
            w.sd256 = state256[i].sd256;
            w.sd256 = _mm256_mul_pd(two256.sd256, w.sd256);
            w.sd256 = _mm256_add_pd(w.sd256, m_three256.sd256);
            paxy[i] = w.sd256;
            w.sd256 = _mm256_mul_pd(w.sd256, w.sd256);
            ar[p] = w.d[0] + w.d[1];
            ar[p + 1] = w.d[2] + w.d[3];
            p += 2;
        }
        _mm256_zeroall();
        p = 0;
        for (int i = 0; i < DSFMTAVX_SIZE * 2; i++) {
            if (ar[i] > 1.0 || ar[i] == 0.0) {
                continue;
            }
            axy[p] = axy[i];
            ar[p] = sqrt(-2.0 * log(ar[i]) / ar[i]);
            p++;
        }
        w256_t cmu;
        w256_t csigma;
        cmu.sd256 = _mm256_set1_pd(mu);
        csigma.sd256 = _mm256_set1_pd(sigma);
        int j = 0;
        for (int i = 0; i < p / 2; i++) {
            __m256d md = _mm256_setr_pd(ar[j], ar[j], ar[j+1], ar[j+1]);
            md = _mm256_mul_pd(md, paxy[i]);
            md = _mm256_mul_pd(md, csigma.sd256);
            w.sd256 = _mm256_add_pd(md, cmu.sd256);
            array256[i].sd256 = w.sd256;
            j += 2;
        }
        if (p % 2 == 1) {
            w128_t w2;
            __m128d md = _mm_set1_pd(ar[p - 1]);
            md = _mm_mul_pd(md, axy[p - 1]);
            md = _mm_mul_pd(md, csigma.sd128[0]);
            w2.sd128 = _mm_add_pd(md, cmu.sd128[0]);
            array[p * 2 - 2] = w2.d[0];
            array[p * 2 - 1] = w2.d[1];
        }
        _mm256_zeroall();
        return p * 2;
    }
#else
    void fillArray256_maxint(double *, int32_t *, uint64_t, int32_t)
    {
        throw new std::logic_error("should not be called");
    }
    int fillArray256_boxmuller(double *, double *, double, double)
    {
        throw new std::logic_error("should not be called");
    }
#endif

#if HAVE_SSE2
    inline __m128i do_uniform128(w128_t *a0, w128_t *b0,
                                 __m128d max128, __m128i min128)
    {
        __m128d a = _mm_add_pd(a0->sd128, m_one128.sd128); // SSE2
        __m128d b = _mm_add_pd(b0->sd128, m_one128.sd128);
        a = _mm_mul_pd(a, max128); // SSE2
        b = _mm_mul_pd(b, max128);
        __m128i y1 = _mm_cvttpd_epi32(a); // SSE2
        __m128i y2 = _mm_cvttpd_epi32(b);
        y2 = _mm_shuffle_epi32(y2, 0x4e); // 0b01001110 // SSE2
        __m128i c = _mm_or_si128(y1, y2); // SSE2
        c = _mm_add_epi32(c, min128);     // SSE2
        return c;
    }

    void fillArray128_maxint(double * state, int32_t *array32,
                             uint64_t rmax, int32_t min)
    {
        DMSG("fillArray128_maxint step 1");
        w128_t * pstate = reinterpret_cast<w128_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        __m128i min128 = _mm_set1_epi32(min); // SSE2
        __m128d max128 = _mm_set1_pd(static_cast<double>(rmax)); //SSE2
        fillState64(state); // there is no fillState128
        DMSG("fillArray128_maxint step 2");
        int j = 0;
        for (int i = 0; i < DSFMTAVX_SIZE * 2; i += 2) {
            array[j++].si128
                = do_uniform128(&pstate[i], &pstate[i + 1], max128, min128);
        }
        fillState64(state); // there is no fillState128
        for (int i = 0; i < DSFMTAVX_SIZE * 2; i += 2) {
            array[j++].si128
                = do_uniform128(&pstate[i], &pstate[i + 1], max128, min128);
        }
    }

    inline void do_boxmuller128(__m128d *axy, double * ar, w128_t *in)
    {
        w128_t w;
        w.sd128 = in->sd128;
        w.sd128 = _mm_mul_pd(two128.sd128, w.sd128); // SSE2
        w.sd128 = _mm_add_pd(w.sd128, m_three128.sd128); // SSE2
        *axy = w.sd128;
        w.sd128 = _mm_mul_pd(w.sd128, w.sd128);
        *ar = w.d[0] + w.d[1];
    }

    inline __m128d do2_boxmuller128(double d, __m128d xy, __m128d csigma,
                                    __m128d cmu)
    {
        __m128d md = _mm_set1_pd(d); // SSE2
        md = _mm_mul_pd(md, xy); // SSE2
        md = _mm_mul_pd(md, csigma);
        return _mm_add_pd(md, cmu); // SSE2
    }

    int fillArray128_boxmuller(double * state, double * array,
                               double mu, double sigma)
    {
        w128_t * state128 = reinterpret_cast<w128_t *>(state);
        w128_t * array128 = reinterpret_cast<w128_t *>(array);
        __m128d axy[DSFMTAVX_SIZE * 2];
        double ar[DSFMTAVX_SIZE * 2];
        fillState64(state); // there is no fillState128
        for (int i = 0; i < DSFMTAVX_SIZE * 2; i++) {
            do_boxmuller128(&axy[i], &ar[i], &state128[i]);
        }
        int p = 0;
        for (int i = 0; i < DSFMTAVX_SIZE * 2; i++) {
            if (ar[i] > 1.0 || ar[i] == 0.0) {
                continue;
            }
            axy[p] = axy[i];
            ar[p] = sqrt(-2.0 * log(ar[i]) / ar[i]);
            p++;
        }
        __m128d cmu = _mm_set1_pd(mu); // SSE2
        __m128d csigma = _mm_set1_pd(sigma);
        for (int i = 0; i < p; i++) {
            array128[i].sd128 = do2_boxmuller128(ar[i], axy[i], csigma, cmu);
        }
        return p * 2;
    }
#else // HAVE_SSE2
    void fillArray128_maxint(double *, int32_t *, uint64_t, int32_t)
    {
        throw new std::logic_error("should not be called");
    }
    int fillArray128_boxmuller(double *, double *, double, double)
    {
        throw new std::logic_error("should not be called");
    }
#endif

    void fillArray64_maxint(double * state, int32_t * array,
                            uint64_t rmax, int32_t min)
    {
        fillState64(state);
        double dmax = static_cast<double>(rmax);
        int j = 0;
        for (int i = 0; i < DSFMTAVX_SIZE * 4; i++) {
            double x = (state[i] - 1.0) * dmax;
            int32_t y = static_cast<int32_t>(x);
            array[j++] = y + min;
        }
        fillState64(state);
        for (int i = 0; i < DSFMTAVX_SIZE * 4; i++) {
            double x = (state[i] - 1.0) * dmax;
            int32_t y = static_cast<int32_t>(x);
            array[j++] = y + min;
        }
    }

    int fillArray64_boxmuller(double * state, double * array,
                              double mu, double sigma)
    {
        w128_t * state128 = reinterpret_cast<w128_t *>(state);
        //w128_t * array128 = reinterpret_cast<w128_t *>(array);
        w128_t w;
        double x;
        double y;
        double r;
        fillState64(state);
        int p = 0;
        for (int i = 0; i < DSFMTAVX_SIZE * 2; i++) {
            w = state128[i];
            x = 2.0 * w.d[0] - 3.0;
            y = 2.0 * w.d[1] - 3.0;
            r = x * x + y * y;
            if (r > 1.0 || r == 0.0) {
                continue;
            }
            double m = sqrt(-2.0 * log(r) / r);
            array[p] = mu + sigma * x * m;
            array[p + 1] = mu + sigma * y * m;
            p += 2;
        }
        return p;
    }
}
/* Local Variables:  */
/* mode: c++         */
/* End:              */
#endif // SFMT_DIST_DSFMT_AVX_COMMON_H
