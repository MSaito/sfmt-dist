#include "sfmt-dist.h"
#include <sfmt-dist/dSFMT19937.h>
#include <sfmt-dist/cpu_feature.h>
#include <sfmt-dist/aligned_alloc.h>

//#define DEBUG

#include <stdexcept>
#include <stdint.h>
#if HAVE_STRING_H
#include <string.h>
#endif

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

#include <cmath>
#include <cfloat>


#if defined(DEBUG)
#include <iostream>
#include <iomanip>
#endif

namespace {
#define DSFMT_N         191
#define DSFMT_MAXDEGREE 19992
#define DSFMT_POS1      117
#define DSFMT_SL1       19
#define DSFMT_SR        12
#define DSFMT_MSK1      UINT64_C(0x000ffafffffffb3f)
#define DSFMT_MSK2      UINT64_C(0x000ffdfffc90fffd)
#define DSFMT_FIX1      UINT64_C(0x90014964b32f4329)
#define DSFMT_FIX2      UINT64_C(0x3b8d12ac548a7c7a)
#define DSFMT_PCV1      UINT64_C(0x3d84e1ac0dc82880)
#define DSFMT_PCV2      UINT64_C(0x0000000000000001)
#define SSE2_SHUFF      0x1b
#define DSFMT_LOW_MASK  UINT64_C(0x000FFFFFFFFFFFFF)
#define DSFMT_HIGH_CONST UINT64_C(0x3FF0000000000000)
#define DSFMT_IDSTR     "dSFMT2-19937:117-19:ffafffffffb3f-ffdfffc90fffd"

    const int size = DSFMT_N;
    const int pos1 = DSFMT_POS1;
    //const int sl1 = DSFMT_SL1;
    //const uint64_t msk1 = DSFMT_MSK1;
    //const uint64_t msk2 = DSFMT_MSK2;
    //const uint64_t parity1 = DSFMT_PCV1;
    //const uint64_t parity2 = DSFMT_PCV1;

    inline static int idxof(int i) {
        return i;
    }

    union w128_t {
        uint64_t u64[2];
        uint32_t u32[4];
        double d[2];
#if HAVE_SSE2
        __m128i si128;
        __m128d sd128;
#endif
    };

#if HAVE_AVX
    union w256x_t {
        double d[4];
        __m256d sd256;
    };
    const w256x_t m_one256 = {{-1, -1, -1, -1}};
#endif

#if HAVE_SSE2
    const w128_t msk128 = {{DSFMT_MSK1, DSFMT_MSK2}};
    const w128_t one = {{1, 1}};
    union w128x_t {
        double d[2];
        __m128d sd128;
    };
    const w128x_t m_one = {{-1, -1}};
    const w128x_t two = {{2, 2}};
    const w128x_t PI2 = {{2 * M_PI, 2 * M_PI}};
    const w128x_t m_two = {{-2, -2}};

    /**
     * This function represents the recursion formula.
     * @param a a 128-bit part of the interal state array
     * @param b a 128-bit part of the interal state array
     * @param c a 128-bit part of the interal state array
     * @param d a 128-bit part of the interal state array
     * @return result
     */
    inline void recursion128(__m128i * r, __m128i * a,
                             __m128i * b, __m128i * u)
    {
        __m128i v, w, x, y, z;
        x = *a;
        z = _mm_slli_epi64(x, DSFMT_SL1);
        y = _mm_shuffle_epi32(*u, SSE2_SHUFF);
        z = _mm_xor_si128(z, *b);
        y = _mm_xor_si128(y, z);

        v = _mm_srli_epi64(y, DSFMT_SR);
        w = _mm_and_si128(y, msk128.si128);
        v = _mm_xor_si128(v, x);
        v = _mm_xor_si128(v, w);
        *r = v;
        *u = y;
    }

    /**
     * This function fills the internal state array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state
     */
    void fillState128(double * state) {
        int i = 0;
        __m128i lung;

        w128_t * pstate = reinterpret_cast<w128_t *>(state);
        lung = pstate[size].si128;
        for (; i < size - pos1; i++) {
            recursion128(&pstate[i].si128, &pstate[i].si128,
                         &pstate[i + pos1].si128, &lung);
        }
        for (; i < size; i++) {
            recursion128(&pstate[i].si128, &pstate[i].si128,
                         &pstate[i + pos1 - size].si128, &lung);
        }
        pstate[size].si128 = lung;
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state.
     * @param array an 128-bit array to be filled by pseudorandom numbers.
     * @param size number of 128-bit pseudorandom numbers to be generated.
     */
    void fillArray128_c1o2(double * state, double * array32, int length)
    {
        DMSG("fillArray128 step 1");
        int i, j;
        w128_t * pstate = reinterpret_cast<w128_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        __m128i lung = pstate[size].si128;
        DMSG("fillArray128 step 2");
        for (i = 0; i < size - pos1; i++) {
            recursion128(&array[i].si128, &pstate[i].si128,
                         &pstate[i + pos1].si128, &lung);
        }
        DMSG("fillArray128 step 3");
        for (; i < size; i++) {
            recursion128(&array[i].si128, &pstate[i].si128,
                         &array[i + pos1 - size].si128, &lung);
        }
        DMSG("fillArray128 step 6");
        for (; i < length; i++) {
            recursion128(&array[i].si128, &array[i - size].si128,
                         &array[i + pos1 - size].si128, &lung);
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            pstate[j++].si128 = array[i].si128;
        }
        pstate[size].si128 = lung;
    }

    inline  void convert128_c0o1(w128_t *w) {
        w->sd128 = _mm_add_pd(w->sd128, m_one.sd128);
    }

    inline  void convert128_o0c1(w128_t *w) {
        w->sd128 = _mm_sub_pd(two.sd128, w->sd128);
    }

    inline  void convert128_o0o1(w128_t *w) {
        w->si128 = _mm_or_si128(w->si128, one.si128);
        w->sd128 = _mm_add_pd(w->sd128, m_one.sd128);
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state.
     * @param array an 128-bit array to be filled by pseudorandom numbers.
     * @param size number of 128-bit pseudorandom numbers to be generated.
     */
    void fillArray128_c0o1(double * state, double * array32, int length)
    {
        DMSG("fillArray128 step 1");
        int i, j;
        w128_t * pstate = reinterpret_cast<w128_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        __m128i lung = pstate[size].si128;
        DMSG("fillArray128 step 2");
        for (i = 0; i < size - pos1; i++) {
            recursion128(&array[i].si128, &pstate[i].si128,
                         &pstate[i + pos1].si128, &lung);
        }
        DMSG("fillArray128 step 3");
        for (; i < size; i++) {
            recursion128(&array[i].si128, &pstate[i].si128,
                         &array[i + pos1 - size].si128, &lung);
        }
        DMSG("fillArray128 step 6");
        for (; i < length; i++) {
            recursion128(&array[i].si128, &array[i - size].si128,
                         &array[i + pos1 - size].si128, &lung);
            convert128_c0o1(&array[i - size]);
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            pstate[j++].si128 = array[i].si128;
            convert128_c0o1(&array[i]);
        }
        pstate[size].si128 = lung;
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state.
     * @param array an 128-bit array to be filled by pseudorandom numbers.
     * @param size number of 128-bit pseudorandom numbers to be generated.
     */
    void fillArray128_o0c1(double * state, double * array32, int length)
    {
        DMSG("fillArray128 step 1");
        int i, j;
        w128_t * pstate = reinterpret_cast<w128_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        __m128i lung = pstate[size].si128;
        DMSG("fillArray128 step 2");
        for (i = 0; i < size - pos1; i++) {
            recursion128(&array[i].si128, &pstate[i].si128,
                         &pstate[i + pos1].si128, &lung);
        }
        DMSG("fillArray128 step 3");
        for (; i < size; i++) {
            recursion128(&array[i].si128, &pstate[i].si128,
                         &array[i + pos1 - size].si128, &lung);
        }
        DMSG("fillArray128 step 6");
        for (; i < length; i++) {
            recursion128(&array[i].si128, &array[i - size].si128,
                         &array[i + pos1 - size].si128, &lung);
            convert128_o0c1(&array[i - size]);
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            pstate[j++].si128 = array[i].si128;
            convert128_o0c1(&array[i]);
        }
        pstate[size].si128 = lung;
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state.
     * @param array an 128-bit array to be filled by pseudorandom numbers.
     * @param size number of 128-bit pseudorandom numbers to be generated.
     */
    void fillArray128_o0o1(double * state, double * array32, int length)
    {
        DMSG("fillArray128 step 1");
        int i, j;
        w128_t * pstate = reinterpret_cast<w128_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        __m128i lung = pstate[size].si128;
        DMSG("fillArray128 step 2");
        for (i = 0; i < size - pos1; i++) {
            recursion128(&array[i].si128, &pstate[i].si128,
                         &pstate[i + pos1].si128, &lung);
        }
        DMSG("fillArray128 step 3");
        for (; i < size; i++) {
            recursion128(&array[i].si128, &pstate[i].si128,
                         &array[i + pos1 - size].si128, &lung);
        }
        DMSG("fillArray128 step 6");
        for (; i < length; i++) {
            recursion128(&array[i].si128, &array[i - size].si128,
                         &array[i + pos1 - size].si128, &lung);
            convert128_o0o1(&array[i - size]);
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            pstate[j++].si128 = array[i].si128;
            convert128_o0o1(&array[i]);
        }
        pstate[size].si128 = lung;
    }

    inline __m128i do_uniform128(__m128d a, __m128d b,
                                 __m128d max128, __m128i min128)
    {
        a = _mm_add_pd(a, m_one.sd128);
        b = _mm_add_pd(b, m_one.sd128);
        a = _mm_mul_pd(a, max128);
        b = _mm_mul_pd(b, max128);
        __m128i y1 = _mm_cvtpd_epi32(a);
        __m128i y2 = _mm_cvtpd_epi32(b);
        y2 = _mm_shuffle_epi32(y2, 0x4e); // 0b01001110
        __m128i c = _mm_or_si128(y1, y2);
        c = _mm_add_epi32(c, min128);
        return c;
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state.
     * @param array an 128-bit array to be filled by pseudorandom numbers.
     * @param size number of 128-bit pseudorandom numbers to be generated.
        uint64_t rmax;
        if (max >= 0 && min >= 0) {
            rmax = max - min;
        } else if (max >= 0 && min < 0) {
            min = -min;
            rmax = max;
            rmax += min;
        } else { // max < 0 && min < 0
            max -= min;
            rmax = max;
        }
     */

    void fillArray128_maxint(double * state, int32_t * array32,
                             uint64_t rmax, int32_t min)
    {
        DMSG("fillArray128_maxint step 1");
        uint32_t mxcsr = _mm_getcsr();
        //xoox xxxx xxxxxxxx
        //1001 f    f   f
        //9fff
        //6000
        //_mm_setcsr((mxcsr & 0xF3FF) | 0x0400);
        _mm_setcsr((mxcsr & 0x9fff) | 0x6000);
        w128_t * pstate = reinterpret_cast<w128_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        __m128i min128 = _mm_set1_epi32(min);
        __m128d max128 = _mm_set1_pd(static_cast<double>(rmax));
        __m128d a;
        __m128d b;
        fillState128(state);
        DMSG("fillArray128 step 2");
        int j = 0;
        for (int i = 0; i < size - 1; i += 2) {
            a = pstate[i].sd128;
            b = pstate[i + 1].sd128;
            array[j++].si128 = do_uniform128(a, b, max128, min128);
        }
        a = pstate[size - 1].sd128;
        fillState128(state);
        for (int i = 0; i < size - 1; i += 2) {
            b = pstate[i].sd128;
            array[j++].si128 = do_uniform128(a, b, max128, min128);
            a = pstate[i + 1].sd128;
        }
        b = pstate[size - 1].sd128;
        array[j++].si128 = do_uniform128(a, b, max128, min128);
        _mm_setcsr(mxcsr);
    }

    int fillArray128_boxmuller(double * state, double * array,
                               double mu, double sigma)
    {
        // size is 191
        w128_t * state128 = reinterpret_cast<w128_t *>(state);
        //w128_t * array128 = reinterpret_cast<w128_t *>(array);
        w128_t w;
        __m128d axy[size];
        __m128d c2 = _mm_set1_pd(2.0);
        __m128d cm3 = _mm_set1_pd(-3.0);
        __m128d cmu = _mm_set1_pd(mu);
        __m128d csigma = _mm_set1_pd(sigma);
        double ar[size];
        fillState128(state);
        for (int i = 0; i < size; i++) {
            w.sd128 = state128[i].sd128;
            w.sd128 = _mm_mul_pd(c2, w.sd128);
            w.sd128 = _mm_add_pd(w.sd128, cm3);
            axy[i] = w.sd128;
            w.sd128 = _mm_mul_pd(w.sd128, w.sd128);
            ar[i] = w.d[0] + w.d[1];
        }
        int p = 0;
        for (int i = 0; i < size; i++) {
            if (ar[i] > 1.0 || ar[i] == 0.0) {
                continue;
            }
            axy[p] = axy[i];
            ar[p] = sqrt(-2.0 * log(ar[i]) / ar[i]);
            p++;
        }
        int j = 0;
        for (int i = 0; i < p; i++) {
            __m128d md = _mm_set1_pd(ar[i]);
            md = _mm_mul_pd(md, axy[i]);
            md = _mm_mul_pd(md, csigma);
            w.sd128 = _mm_add_pd(md, cmu);
            array[j] = w.d[0];
            array[j + 1] = w.d[1];
            j += 2;
        }
        return p * 2;
    }

#else
    void fillState128(double *)
    {
        throw new std::logic_error("should not be called");
    }

    void fillArray128_c1o2(double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray128_c0o1(double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray128_o0c1(double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray128_o0o1(double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray128_maxint(double *, int32_t *, uint64_t, int32_t)
    {
        throw new std::logic_error("should not be called");
    }
#endif // HAVE_SSE2

#if HAVE_AVX
    // get 256 and return 128
    inline __m128i do_uniform256(__m256d a, __m256d max256, __m128i min128)
    {
        a = _mm256_add_pd(a, m_one256.sd256);
        a = _mm256_mul_pd(a, max256);
        __m128i y = _mm256_cvtpd_epi32(a);
        return _mm_add_epi32(y, min128);
    }

    void fillArray256_maxint(double * state, int32_t * array32,
                             uint64_t rmax, int32_t min)
    {
        union w256_t {
            uint64_t u64[4];
            uint32_t u32[8];
            double d[4];
            __m256i si256;
            __m256d sd256;
        };
        DMSG("fillArray256_maxint step 1");
        uint32_t mxcsr = _mm_getcsr();
        _mm_setcsr((mxcsr & 0x9fff) | 0x6000);
        w128_t * pstate = reinterpret_cast<w128_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        __m128i min128 = _mm_set1_epi32(min);
        __m256d max256 = _mm256_set1_pd(static_cast<double>(rmax));
        __m256d a;
        fillState128(state);
        DMSG("fillArray256_maxint step 2");
        int j = 0;
        for (int i = 0; i < size - 1; i += 2) {
            a = _mm256_load_pd(reinterpret_cast<double *>(&pstate[i].sd128));
            array[j++].si128 = do_uniform256(a, max256, min128);
        }
        __m128d a1 = pstate[size - 1].sd128;
        fillState128(state);
        __m128d a2 = pstate[0].sd128;
        //a = _mm256_set_m128d(a2, a1); // can not compile
        a = _mm256_insertf128_pd(a, a1, 0);
        a = _mm256_insertf128_pd(a, a2, 1);
        array[j++].si128 = do_uniform256(a, max256, min128);
        int start = 1;
        for (int i = start; i < size; i += 2) {
#if 0
            // loadu が遅いのか。
            a = _mm256_loadu_pd(reinterpret_cast<double *>(&pstate[i].sd128));
#else
            a1 = pstate[i].sd128;
            a2 = pstate[i + 1].sd128;
            a = _mm256_insertf128_pd(a, a1, 0);
            a = _mm256_insertf128_pd(a, a2, 1);
#endif
            array[j++].si128 = do_uniform256(a, max256, min128);
        }
        _mm_setcsr(mxcsr);
        _mm256_zeroall();
    }
#else
    void fillArray256_maxint(double *, int32_t *, uint64_t, int32_t)
    {
        throw new std::logic_error("should not be called");
    }
#endif

    /**
     * This function represents the recursion formula.
     * @param r output
     * @param a a 128-bit part of the internal state array
     * @param b a 128-bit part of the internal state array
     * @param c a 128-bit part of the internal state array
     * @param d a 128-bit part of the internal state array
     */
    inline static void recursion64(w128_t *r, w128_t *a, w128_t * b,
                                   w128_t *lung)
    {
        uint64_t t0;
        uint64_t t1;
        uint64_t L0;
        uint64_t L1;

        t0 = a->u64[0];
        t1 = a->u64[1];
        L0 = lung->u64[0];
        L1 = lung->u64[1];
        lung->u64[0] = (t0 << DSFMT_SL1) ^ (L1 >> 32) ^ (L1 << 32) ^ b->u64[0];
        lung->u64[1] = (t1 << DSFMT_SL1) ^ (L0 >> 32) ^ (L0 << 32) ^ b->u64[1];
        r->u64[0] = (lung->u64[0] >> DSFMT_SR) ^ (lung->u64[0] & DSFMT_MSK1)
            ^ t0;
        r->u64[1] = (lung->u64[1] >> DSFMT_SR) ^ (lung->u64[1] & DSFMT_MSK2)
            ^ t1;
    }

    inline  void convert64_c0o1(w128_t *w) {
        w->d[0] -= 1.0;
        w->d[1] -= 1.0;
    }

    inline  void convert64_o0c1(w128_t *w) {
        w->d[0] = 2.0 - w->d[0];
        w->d[1] = 2.0 - w->d[1];
    }

    inline  void convert64_o0o1(w128_t *w) {
        w->u64[0] |= 1;
        w->u64[1] |= 1;
        w->d[0] -= 1.0;
        w->d[1] -= 1.0;
    }

    /**
     * This function fills the internal state array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state
     */
    void fillState64(double * state64)
    {
        int i;
        w128_t * state = reinterpret_cast<w128_t *>(state64);
        w128_t lung = state[size];
        for (i = 0; i < size - pos1; i++) {
            recursion64(&state[i], &state[i],
                        &state[i + pos1], &lung);
        }
        for (; i < size; i++) {
            recursion64(&state[i], &state[i],
                        &state[i + pos1 - size], &lung);
        }
        state[size] = lung;
    }

    void fillArray64_c1o2(double * state64, double * array64, int length) {
        int i, j;
        DMSG("start fillArray32");

        w128_t * state = reinterpret_cast<w128_t *>(state64);
        w128_t * array = reinterpret_cast<w128_t *>(array64);
        w128_t lung = state[size];
        for (i = 0; i < size - pos1; i++) {
            recursion64(&array[i], &state[i], &state[i + pos1], &lung);
        }
        for (; i < size; i++) {
            recursion64(&array[i], &state[i], &array[i + pos1 - size], &lung);
        }
        for (; i < length; i++) {
            recursion64(&array[i], &array[i - size],
                        &array[i + pos1 - size], &lung);
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            state[j++] = array[i];
        }
        state[size] = lung;
    }

    void fillArray64_c0o1(double * state64, double * array64, int length) {
        int i, j;
        DMSG("start fillArray32");

        w128_t * state = reinterpret_cast<w128_t *>(state64);
        w128_t * array = reinterpret_cast<w128_t *>(array64);
        w128_t lung = state[size];
        for (i = 0; i < size - pos1; i++) {
            recursion64(&array[i], &state[i], &state[i + pos1], &lung);
        }
        for (; i < size; i++) {
            recursion64(&array[i], &state[i], &array[i + pos1 - size], &lung);
        }
        for (; i < length; i++) {
            recursion64(&array[i], &array[i - size],
                        &array[i + pos1 - size], &lung);
            convert64_c0o1(&array[i - size]);
        }
        j = 0;
        for (int i = length - size; i < length; i++) {
            state[j++] = array[i];
            convert64_c0o1(&array[i]);
        }
        state[size] = lung;
    }

    void fillArray64_o0c1(double * state64, double * array64, int length) {
        int i, j;
        DMSG("start fillArray32");

        w128_t * state = reinterpret_cast<w128_t *>(state64);
        w128_t * array = reinterpret_cast<w128_t *>(array64);
        w128_t lung = state[size];
        for (i = 0; i < size - pos1; i++) {
            recursion64(&array[i], &state[i], &state[i + pos1], &lung);
        }
        for (; i < size; i++) {
            recursion64(&array[i], &state[i], &array[i + pos1 - size], &lung);
        }
        for (; i < length; i++) {
            recursion64(&array[i], &array[i - size],
                        &array[i + pos1 - size], &lung);
            convert64_o0c1(&array[i - size]);
        }
        j = 0;
        for (int i = length - size; i < length; i++) {
            state[j++] = array[i];
            convert64_o0c1(&array[i]);
        }
        state[size] = lung;
    }

    void fillArray64_o0o1(double * state64, double * array64, int length) {
        int i, j;
        DMSG("start fillArray32");

        w128_t * state = reinterpret_cast<w128_t *>(state64);
        w128_t * array = reinterpret_cast<w128_t *>(array64);
        w128_t lung = state[size];
        for (i = 0; i < size - pos1; i++) {
            recursion64(&array[i], &state[i], &state[i + pos1], &lung);
        }
        for (; i < size; i++) {
            recursion64(&array[i], &state[i], &array[i + pos1 - size], &lung);
        }
        for (; i < length; i++) {
            recursion64(&array[i], &array[i - size],
                        &array[i + pos1 - size], &lung);
            convert64_o0o1(&array[i - size]);
        }
        j = 0;
        for (int i = length - size; i < length; i++) {
            state[j++] = array[i];
            convert64_o0o1(&array[i]);
        }
        state[size] = lung;
    }

    void fillArray64_maxint(double * state, int32_t * array,
                            uint64_t rmax, int32_t min)
    {
        fillState64(state);
        double dmax = static_cast<double>(rmax);
        int j = 0;
        for (int i = 0; i < size * 2; i++) {
            double x = (state[i] - 1.0) * dmax;
            int32_t y = static_cast<int32_t>(x);
            array[j++] = y + min;
        }
        fillState64(state);
        for (int i = 0; i < size * 2; i++) {
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
        for (int i = 0; i < size; i++) {
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
    static void period_certification(w128_t * state) {
        uint64_t pcv[2] = {DSFMT_PCV1, DSFMT_PCV2};
        uint64_t tmp[2];
        uint64_t inner;
        tmp[0] = (state[size].u64[0] ^ DSFMT_FIX1);
        tmp[1] = (state[size].u64[1] ^ DSFMT_FIX2);

        inner = tmp[0] & pcv[0];
        inner ^= tmp[1] & pcv[1];
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
#if (DSFMT_PCV2 & 1) == 1
        state[size].u64[1] ^= 1;
#else
        int i;
        int j;
        uint64_t work;
        for (i = 1; i >= 0; i--) {
            work = 1;
            for (j = 0; j < 64; j++) {
                if ((work & pcv[i]) != 0) {
                    state[size].u64[i] ^= work;
                    return;
                }
                work = work << 1;
            }
        }
#endif
        return;
    }

    /**
     * This function initializes the internal state array to fit the IEEE
     * 754 format.
     * @param dsfmt dsfmt state vector.
     */
    static void initial_mask(w128_t * state) {
        int i;
        uint64_t *psfmt;

        psfmt = &state[0].u64[0];
        for (i = 0; i < size * 2; i++) {
            psfmt[i] = (psfmt[i] & DSFMT_LOW_MASK) | DSFMT_HIGH_CONST;
        }
    }

    /**
     * This function initializes the internal state array with a 32-bit
     * integer seed.
     * @param dsfmt dsfmt state vector.
     * @param seed a 32-bit integer used as the seed.
     * @param mexp caller's mersenne expornent
     */
    void init(w128_t * state, uint32_t seed) {
        int i;
        uint32_t *psfmt;

        psfmt = &state[0].u32[0];
        psfmt[idxof(0)] = seed;
        for (i = 1; i < (DSFMT_N + 1) * 4; i++) {
            psfmt[idxof(i)] = 1812433253UL
                * (psfmt[idxof(i - 1)] ^ (psfmt[idxof(i - 1)] >> 30)) + i;
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
    void init(w128_t * state, uint32_t init_key[], int key_length) {
        int i, j, count;
        uint32_t r;
        uint32_t *psfmt32;
        int lag;
        int mid;
        int size = (DSFMT_N + 1) * 4;   /* pulmonary */

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
        r = ini_func1(psfmt32[idxof(0)] ^ psfmt32[idxof(mid % size)]
                      ^ psfmt32[idxof((size - 1) % size)]);
        psfmt32[idxof(mid % size)] += r;
        r += key_length;
        psfmt32[idxof((mid + lag) % size)] += r;
        psfmt32[idxof(0)] = r;
        count--;
        for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
            r = ini_func1(psfmt32[idxof(i)]
                          ^ psfmt32[idxof((i + mid) % size)]
                          ^ psfmt32[idxof((i + size - 1) % size)]);
            psfmt32[idxof((i + mid) % size)] += r;
            r += init_key[j] + i;
            psfmt32[idxof((i + mid + lag) % size)] += r;
            psfmt32[idxof(i)] = r;
            i = (i + 1) % size;
        }
        for (; j < count; j++) {
            r = ini_func1(psfmt32[idxof(i)]
                          ^ psfmt32[idxof((i + mid) % size)]
                          ^ psfmt32[idxof((i + size - 1) % size)]);
            psfmt32[idxof((i + mid) % size)] += r;
            r += i;
            psfmt32[idxof((i + mid + lag) % size)] += r;
            psfmt32[idxof(i)] = r;
            i = (i + 1) % size;
        }
        for (j = 0; j < size; j++) {
            r = ini_func2(psfmt32[idxof(i)]
                          + psfmt32[idxof((i + mid) % size)]
                          + psfmt32[idxof((i + size - 1) % size)]);
            psfmt32[idxof((i + mid) % size)] ^= r;
            r -= i;
            psfmt32[idxof((i + mid + lag) % size)] ^= r;
            psfmt32[idxof(i)] = r;
            i = (i + 1) % size;
        }
        initial_mask(state);
        period_certification(state);
    }
}

namespace MersenneTwister {
    using namespace std;

    DSFMT19937::DSFMT19937(uint32_t seedValue)
    {
        cf = cpu_feature();
        size_t alloc_size = (size + 1) * 2 * sizeof(uint64_t);
        state = alignedAlloc<double *>(alloc_size);
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue);
    }

    DSFMT19937::DSFMT19937(uint32_t seedValue[], int length)
    {
        cf = cpu_feature();
        size_t alloc_size = (size + 1) * 2 * sizeof(uint64_t);
        state = alignedAlloc<double *>(alloc_size);
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue, length);
    }

    DSFMT19937::~DSFMT19937()
    {
        alignedFree(state);
    }

    const char * DSFMT19937::getIDString()
    {
        return DSFMT_IDSTR;
    }

    /**
     * This function initializes the internal state array with a 32-bit
     * integer seed.
     *
     * @param sfmt SFMT internal state
     * @param seed a 32-bit integer used as the seed.
     */
    void DSFMT19937::seed(uint32_t seedValue)
    {
        w128_t * state128 = reinterpret_cast<w128_t *>(state);
        init(state128, seedValue);
        index = stateSize64;
    }

    /**
     * This function initializes the internal state array,
     * with an array of 32-bit integers used as the seeds
     * @param sfmt SFMT internal state
     * @param init_key the array of 32-bit integers, used as a seed.
     * @param key_length the length of init_key.
     */
    void DSFMT19937::seed(uint32_t *init_key, int key_length)
    {
        w128_t * state128 = reinterpret_cast<w128_t *>(state);
        init(state128, init_key, key_length);
        index = stateSize64;
    }

    void DSFMT19937::fillState()
    {
        if (cf.sse2) {
            fillState128(state);
        } else {
            fillState64(state);
        }
        index = 0;
    }

    void DSFMT19937::fillArray(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 2 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generate();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 16;
        if ((align == 0) && cf.sse2) {
            DMSG("fillArray step 3");
            fillArray128_c0o1(state, array, length / 2);
        } else {
            DMSG("fillArray step 4");
            fillArray64_c0o1(state, array, length / 2);
        }
    }

    void DSFMT19937::fillArrayClose1Open2(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 2 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateClose1Open2();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 16;
        if ((align == 0) && cf.sse2) {
            DMSG("fillArray step 3");
            fillArray128_c1o2(state, array, length / 2);
        } else {
            DMSG("fillArray step 4");
            fillArray64_c1o2(state, array, length / 2);
        }
    }

    void DSFMT19937::fillArrayOpen0Close1(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 2 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateOpen0Close1();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 16;
        if ((align == 0) && cf.sse2) {
            DMSG("fillArray step 3");
            fillArray128_o0c1(state, array, length / 2);
        } else {
            DMSG("fillArray step 4");
            fillArray64_o0c1(state, array, length / 2);
        }
    }

    void DSFMT19937::fillArrayOpen0Open1(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 2 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateOpen0Open1();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 16;
        if ((align == 0) && cf.sse2) {
            DMSG("fillArray step 3");
            fillArray128_o0o1(state, array, length / 2);
        } else {
            DMSG("fillArray step 4");
            fillArray64_o0o1(state, array, length / 2);
        }
    }

    /*
     * array の要素数はstateの要素数の2倍
     */
    void DSFMT19937::fillArrayMaxInt(int32_t * array,
                                     uint64_t rmax, int32_t min)
    {
#if 0
        fillArray64_maxint(state, array, rmax, min);
#else
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx) {
            fillArray256_maxint(state, array, rmax, min);
        } else if ((align % 16 == 0) && cf.sse2) {
            fillArray128_maxint(state, array, rmax, min);
        } else {
            fillArray64_maxint(state, array, rmax, min);
        }
#endif
    }

    /*
     * array の要素数はstateの要素数と同じ
     */
    int DSFMT19937::fillArrayNormalDist(double * array,
                                       double mu, double sigma)
    {
        if (cf.sse2) {
            return fillArray128_boxmuller(state, array, mu, sigma);
        } else {
            return fillArray64_boxmuller(state, array, mu, sigma);
        }
    }

    bool DSFMT19937::selfTest()
    {
        DMSG("selfTest Start");
        DMSG("fillArray_maxInt test Start");
        if (cf.avx) {
            DMSG("fillArray_maxInt AVX test Start");
            int asize = blockSize() * 2;
            int32_t * array1 = alignedAlloc<int32_t *>(asize * sizeof(double));
            int32_t * array2 = alignedAlloc<int32_t *>(asize * sizeof(double));
            seed(0);
            fillArray64_maxint(state, array1, 200, 1);
            seed(0);
            fillArray256_maxint(state, array2, 200, 1);
            for (int i = 0; i < asize; i++) {
                if (fabs(array1[i] - array2[i]) >= DBL_EPSILON) {
                    DMSG("fillArray_maxInt AVX test NG array mismatch");
#if defined(DEBUG)
                    std::cout << "i = " << std::dec << i << std::endl;
#endif
                    return false;
                }
            }
        }
        if (cf.sse2) {
            DMSG("fillArray_maxInt SSE2 test Start");
            int asize = blockSize() * 2;
            int32_t * array1 = alignedAlloc<int32_t *>(asize * sizeof(double));
            int32_t * array2 = alignedAlloc<int32_t *>(asize * sizeof(double));
            seed(0);
            fillArray64_maxint(state, array1, 200, 1);
            seed(0);
            fillArray128_maxint(state, array2, 200, 1);
            for (int i = 0; i < asize; i++) {
                if (fabs(array1[i] - array2[i]) >= DBL_EPSILON) {
                    DMSG("fillArray_maxInt SSE2 test NG array mismatch");
#if defined(DEBUG)
                    std::cout << "i = " << std::dec << i << std::endl;
#endif
                    return false;
                }
            }
        }
        DMSG("fillArray_maxInt test OK");
        DMSG("fillArray_boxmuller test Start");
        if (cf.sse2) {
            int asize = blockSize();
            double * array1 = alignedAlloc<double *>(asize * sizeof(double));
            double * array2 = alignedAlloc<double *>(asize * sizeof(double));
            seed(0);
            int c1 = fillArray64_boxmuller(state, array1, 0, 1.0);
            seed(0);
            int c2 = fillArray128_boxmuller(state, array2, 0, 1.0);
            if (c1 != c2) {
                DMSG("fillArray_boxmuller test NG c1 != c2");
                return false;
            }
            for (int i = 0; i < asize; i++) {
                if (fabs(array1[i] - array2[i]) >= DBL_EPSILON) {
                    DMSG("fillArray_boxmuller test NG array mismatch");
#if defined(DEBUG)
                    std::cout << "i = " << std::dec << i << std::endl;
#endif
                    return false;
                }
            }
            DMSG("fillArray_boxmuller test OK");
        }
        return true;
    }

}
