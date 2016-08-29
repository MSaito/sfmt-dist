#include "sfmt-dist.h"
#include <sfmt-dist/dSFMTAVX607.h>
#include <sfmt-dist/cpu_feature.h>
#include <sfmt-dist/aligned_alloc.h>
#include "w256.h"
#include "w128.h"

#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdint.h>
#include <cmath>
#include <cfloat>

#if HAVE_STRING_H
#include <string.h>
#endif
#if HAVE_MEMORY_H
#include <memory.h>
#endif

#if HAVE_ZMMINTRIN_H
#include <zmmintrin.h>
#endif

#if HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif

namespace {
    using namespace MersenneTwister;
    using namespace std;

    void fillState64(double * state64);

#define DSFMTAVX2_SL1 19
#define DSFMTAVX2_SR 12

    const uint64_t low_mask = UINT64_C(0x000FFFFFFFFFFFFF);
    const uint64_t high_const = UINT64_C(0x3FF0000000000000);
    const w256_t mask1 = {{UINT64_C(0x000f7fdedfdaf6f7),
                           UINT64_C(0x000ff71b5676ff7f),
                           UINT64_C(0x000fb66b7a1ef925),
                           UINT64_C(0x000bfaffffff7faf)}};
    const w256_t fix1 = {{UINT64_C(0x46f5e1807e0eebd9),
                          UINT64_C(0x206a072aabbcf1ca),
                          UINT64_C(0xca0610e8704e7f0e),
                          UINT64_C(0xba564f522afc7696)}};
    const w256_t pcv1 = {{UINT64_C(0x0000000000000001),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0x4000000000000000),
                          UINT64_C(0xb9f61e13e9f75e26)}};
    const int array_size = 2;
    const int pos1 = 1;

#if HAVE_SSE2
    const w128_t one128 = {{1, 1}};
    const w128xd_t m_one128 = {{-1, -1}};
    const w128xd_t two128 = {{2, 2}};
    const w128xd_t PI2128 = {{2 * M_PI, 2 * M_PI}};
    const w128xd_t m_two128 = {{-2, -2}};
    const w128xd_t m_three128 = {{-3.0, -3.0}};
#endif

    string getIDString()
    {
        stringstream ss;
        char delim = ':';
        ss << "dSFMTAVX-607:1-19";
        //for (int i = 3; i >= 0; i--) {
        for (int i = 0; i < 4; i++) {
            //ss << delim << setfill('0') << setw(16) << hex << mask1.u64[i];
            ss << delim << setfill('0') << setw(13) << hex << mask1.u64[i];
            delim = '-';
        }
        string s;
        ss >> s;
        return s;
    }
#if HAVE_SSE2
    inline __m128i do_uniform128(w128_t *a0, w128_t *b0,
                                 __m128d max128, __m128i min128)
    {
        __m128d a = _mm_add_pd(a0->sd128, m_one128.sd128);
        __m128d b = _mm_add_pd(b0->sd128, m_one128.sd128);
        a = _mm_mul_pd(a, max128);
        b = _mm_mul_pd(b, max128);
        __m128i y1 = _mm_cvtpd_epi32(a);
        __m128i y2 = _mm_cvtpd_epi32(b);
        y2 = _mm_shuffle_epi32(y2, 0x4e); // 0b01001110
        __m128i c = _mm_or_si128(y1, y2);
        c = _mm_add_epi32(c, min128);
        return c;
    }

    void fillArray128_maxint(double * state, int32_t * array32,
                             uint64_t rmax, int32_t min)
    {
        DMSG("fillArray128_maxint step 1");
        uint32_t mxcsr = _mm_getcsr();
        _mm_setcsr((mxcsr & 0x9fff) | 0x6000); // truncate to zero
        w128_t * pstate = reinterpret_cast<w128_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        __m128i min128 = _mm_set1_epi32(min);
        __m128d max128 = _mm_set1_pd(static_cast<double>(rmax));
        fillState64(state);
        DMSG("fillArray128_maxint step 2");
        array[0].si128 = do_uniform128(&pstate[0], &pstate[1], max128, min128);
        array[1].si128 = do_uniform128(&pstate[2], &pstate[3], max128, min128);
        fillState64(state);
        array[2].si128 = do_uniform128(&pstate[0], &pstate[1], max128, min128);
        array[3].si128 = do_uniform128(&pstate[2], &pstate[3], max128, min128);
        _mm_setcsr(mxcsr);
    }

    inline void do_boxmuller128(__m128d *axy, double * ar, w128_t *in,
                                __m128d c2, __m128d cm3)
    {
        w128_t w;
        w.sd128 = in->sd128;
        w.sd128 = _mm_mul_pd(c2, w.sd128);
        w.sd128 = _mm_add_pd(w.sd128, cm3);
        *axy = w.sd128;
        w.sd128 = _mm_mul_pd(w.sd128, w.sd128);
        *ar = w.d[0] + w.d[1];
    }

    inline __m128d do2_boxmuller128(double d, __m128d xy, __m128d csigma,
                                    __m128d cmu)
    {
        __m128d md = _mm_set1_pd(d);
        md = _mm_mul_pd(md, xy);
        md = _mm_mul_pd(md, csigma);
        return _mm_add_pd(md, cmu);
    }

    int fillArray128_boxmuller(double * state, double * array,
                               double mu, double sigma)
    {
        // array_size is 2
        w128_t * state128 = reinterpret_cast<w128_t *>(state);
        w128_t * array128 = reinterpret_cast<w128_t *>(array);
        //w128_t w;
        __m128d axy[array_size * 2];
        //__m128d c2 = _mm_set1_pd(2.0);
        __m128d c2 = two128.sd128;
        //__m128d cm3 = _mm_set1_pd(-3.0);
        __m128d cm3 = m_three128.sd128;
        __m128d cmu = _mm_set1_pd(mu);
        __m128d csigma = _mm_set1_pd(sigma);
        double ar[array_size * 2];
        fillState64(state);
        do_boxmuller128(&axy[0], &ar[0], &state128[0], c2, cm3);
        do_boxmuller128(&axy[1], &ar[1], &state128[1], c2, cm3);
        do_boxmuller128(&axy[2], &ar[2], &state128[2], c2, cm3);
        do_boxmuller128(&axy[3], &ar[3], &state128[3], c2, cm3);
        int p = 0;
        for (int i = 0; i < array_size * 2; i++) {
            if (ar[i] > 1.0 || ar[i] == 0.0) {
                continue;
            }
            axy[p] = axy[i];
            ar[p] = sqrt(-2.0 * log(ar[i]) / ar[i]);
            p++;
        }
        switch (p) {
        case 4:
            array128[0].sd128 = do2_boxmuller128(ar[0], axy[0], csigma, cmu);
            array128[1].sd128 = do2_boxmuller128(ar[1], axy[1], csigma, cmu);
            array128[2].sd128 = do2_boxmuller128(ar[2], axy[2], csigma, cmu);
            array128[3].sd128 = do2_boxmuller128(ar[3], axy[3], csigma, cmu);
            return p * 2;
        case 3:
            array128[0].sd128 = do2_boxmuller128(ar[0], axy[0], csigma, cmu);
            array128[1].sd128 = do2_boxmuller128(ar[1], axy[1], csigma, cmu);
            array128[2].sd128 = do2_boxmuller128(ar[2], axy[2], csigma, cmu);
            return p * 2;
        case 2:
            array128[0].sd128 = do2_boxmuller128(ar[0], axy[0], csigma, cmu);
            array128[1].sd128 = do2_boxmuller128(ar[1], axy[1], csigma, cmu);
            return p * 2;
        case 1:
            array128[0].sd128 = do2_boxmuller128(ar[0], axy[0], csigma, cmu);
            return p * 2;
        case 0:
        default:
            return 0;
        }
    }
#else
    void fillArray128_maxint(double *, int32_t *, uint64_t, int32_t)
    {
        throw new std::logic_error("should not be called");
    }
    int fillArray128_boxmuller(double *, double *, double, double)
    {
        throw new std::logic_error("should not be called");
    }
#endif

#if HAVE_AVX2 && HAVE_IMMINTRIN_H
    const w256x32_t perm = {{7, 0, 1, 2, 3, 4, 5, 6}};
    const w256_t one = {{1, 1, 1, 1}};
    const w256xd_t m_one = {{-1.0, -1.0, -1.0, -1.0}};
    const w256xd_t two = {{2.0, 2.0, 2.0, 2.0}};

    /**
     * This function represents the recursion formula.
     * @param params parameters
     * @param a a 128-bit part of the interal state array
     * @param b a 128-bit part of the interal state array
     * @param c a 128-bit part of the interal state array
     * @param d a 128-bit part of the interal state array
     * @return result
     */
    inline __m256i recursion256(__m256i a, __m256i b, __m256i * u)
    {
        __m256i x = _mm256_slli_epi64(a, DSFMTAVX2_SL1);
        __m256i y = _mm256_permutevar8x32_epi32(*u, perm.si256);
        __m256i z = _mm256_xor_si256(x, b);
        y = _mm256_xor_si256(y, z);
        __m256i v = _mm256_srli_epi64(y, DSFMTAVX2_SR);
        __m256i w = _mm256_and_si256(y, mask1.si256);
        __m256i s = _mm256_xor_si256(a, v);
        *u = y;
        return _mm256_xor_si256(w, s);
    }

    /**
     * This function fills the internal state array with pseudorandom
     * integers.
     * @param p parameter
     * @param state SFMT internal state
     */
    void fillState256(double * state) {


        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        __m256i lung = pstate[2].si256;
        pstate[0].si256 = recursion256(pstate[0].si256,
                                       pstate[1].si256, &lung);
        pstate[1].si256 = recursion256(pstate[1].si256,
                                       pstate[0].si256, &lung);
        pstate[2].si256 = lung;
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
        DMSG("fillArray256_c1o2 step 1");
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        __m256i a = pstate[0].si256;
        __m256i b = pstate[1].si256;
        __m256i lung = pstate[2].si256;
        __m256i t;
        for (int i = 0; i < length; i++) {
            t = recursion256(a, b, &lung);
            array[i].si256 = t;
            a = b;
            b = t;
        }
        pstate[0].si256 = a;
        pstate[1].si256 = b;
        pstate[2].si256 = lung;
        _mm256_zeroall();
    }

    inline  __m256d convert256_c0o1(__m256i w) {
        return _mm256_add_pd((__m256d)w, m_one.sd256);
    }

    inline  __m256d convert256_o0c1(__m256i w) {
        return _mm256_sub_pd(two.sd256, (__m256d)w);
    }

    inline  __m256d convert256_o0o1(__m256i w) {
        w = _mm256_or_si256(w, one.si256);
        return _mm256_add_pd((__m256d)w, m_one.sd256);
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param p parameter
     * @param state SFMT internal state.
     * @param array64 an 256-bit array to be filled by pseudorandom numbers.
     * @param length number of 256-bit pseudorandom numbers to be generated.
     */
    void fillArray256_c0o1(double * state,
                           double * array64, int length)
    {
        DMSG("fillArray256_c0o1 step 1");
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        __m256i a = pstate[0].si256;
        __m256i b = pstate[1].si256;
        __m256i lung = pstate[2].si256;
        __m256i t;
        for (int i = 0; i < length; i++) {
            t = recursion256(a, b, &lung);
            array[i].sd256 = convert256_c0o1(t);
            a = b;
            b = t;
        }
        pstate[0].si256 = a;
        pstate[1].si256 = b;
        pstate[2].si256 = lung;
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
    void fillArray256_o0c1(double * state,
                           double * array64, int length)
    {
        DMSG("fillArray256_o0c1 step 1");
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        __m256i a = pstate[0].si256;
        __m256i b = pstate[1].si256;
        __m256i lung = pstate[2].si256;
        __m256i t;
        for (int i = 0; i < length; i++) {
            t = recursion256(a, b, &lung);
            array[i].sd256 = convert256_o0c1(t);
            a = b;
            b = t;
        }
        pstate[0].si256 = a;
        pstate[1].si256 = b;
        pstate[2].si256 = lung;
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
    void fillArray256_o0o1(double * state,
                           double * array64, int length)
    {
        DMSG("fillArray256 step 1");
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        __m256i a = pstate[0].si256;
        __m256i b = pstate[1].si256;
        __m256i lung = pstate[2].si256;
        __m256i t;
        for (int i = 0; i < length; i++) {
            t = recursion256(a, b, &lung);
            array[i].sd256 = convert256_o0o1(t);
            a = b;
            b = t;
        }
        pstate[0].si256 = a;
        pstate[1].si256 = b;
        pstate[2].si256 = lung;
        _mm256_zeroall();
    }

    inline __m128i do_uniform256(w256_t *a0, __m256d max256, __m128i min128)
    {
        __m256d a = _mm256_add_pd(a0->sd256, m_one.sd256);
        a = _mm256_mul_pd(a, max256);
        __m128i y = _mm256_cvtpd_epi32(a);
        return  _mm_add_epi32(y, min128);
    }

    void fillArray256_maxint(double * state, int32_t * array32,
                             uint64_t rmax, int32_t min)
    {
        DMSG("fillArray128_maxint step 1");
        uint32_t mxcsr = _mm_getcsr();
        _mm_setcsr((mxcsr & 0x9fff) | 0x6000); // truncate to zero
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        __m128i min128 = _mm_set1_epi32(min);
        __m256d max256 = _mm256_set1_pd(static_cast<double>(rmax));
        fillState256(state);
        DMSG("fillArray128_maxint step 2");
        //int j = 0;
        array[0].si128 = do_uniform256(&pstate[0], max256, min128);
        array[1].si128 = do_uniform256(&pstate[1], max256, min128);
        fillState256(state);
        array[2].si128 = do_uniform256(&pstate[0], max256, min128);
        array[3].si128 = do_uniform256(&pstate[1], max256, min128);
        _mm_setcsr(mxcsr);
        _mm256_zeroall();
    }
#if 0
//    inline void do_boxmuller256(__m256d *axy, double * ar, w256_t *in,
//                                __m256d c2, __m256d cm3, __m256d zero)
    inline void do_boxmuller256(__m256d *axy, double * ar, w256_t *in,
                                __m256d c2, __m256d cm3)
    {
        w256_t w;
        w.sd256 = in->sd256;
        w.sd256 = _mm256_mul_pd(c2, w.sd256);
        w.sd256 = _mm256_add_pd(w.sd256, cm3);
        *axy = w.sd256;
        w.sd256 = _mm256_mul_pd(w.sd256, w.sd256);
        // _mm256_hadd_pd 水平加算
        ar[0] = w.d[0] + w.d[1];
        ar[1] = w.d[2] + w.d[3];
    }

    inline __m256d do2_boxmuller256(double d, __m256d xy, __m256d csigma,
                                    __m256d cmu)
    {
        __m256d md = _mm256_set1_pd(d);
        md = _mm256_mul_pd(md, xy);
        md = _mm256_mul_pd(md, csigma);
        return _mm256_add_pd(md, cmu);
    }

    int fillArray256_boxmuller(double * state, double * array,
                               double mu, double sigma)
    {
        // array_size is 2
        w256_t * state256 = reinterpret_cast<w256_t *>(state);
        w128_t * array128 = reinterpret_cast<w128_t *>(array);
        w256_t w;
        __m256d axy[array_size * 2];
        __m256d c2 = _mm256_set1_pd(2.0);
        __m256d cm3 = _mm256_set1_pd(-3.0);
        __m128d cmu = _mm_set1_pd(mu);
        __m128d csigma = _mm_set1_pd(sigma);
        double ar[array_size * 2];
        fillState256(state);
        do_boxmuller256(&axy[0], &ar[0], &state256[0], c2, cm3);
        do_boxmuller256(&axy[1], &ar[2], &state256[1], c2, cm3);
        int p = 0;
        for (int i = 0; i < array_size * 2; i++) {
            if (ar[i] > 1.0 || ar[i] == 0.0) {
                continue;
            }
            axy[p] = axy[i];
            ar[p] = sqrt(-2.0 * log(ar[i]) / ar[i]);
            p++;
        }
        //ここは128でいい
        switch (p) {
        case 4:
            array128[0].sd128 = do2_boxmuller128(ar[0], axy[0], csigma, cmu);
            array128[1].sd128 = do2_boxmuller128(ar[1], axy[1], csigma, cmu);
            array128[2].sd128 = do2_boxmuller128(ar[2], axy[2], csigma, cmu);
            array128[3].sd128 = do2_boxmuller128(ar[3], axy[3], csigma, cmu);
            return p * 2;
        case 3:
            array128[0].sd128 = do2_boxmuller128(ar[0], axy[0], csigma, cmu);
            array128[1].sd128 = do2_boxmuller128(ar[1], axy[1], csigma, cmu);
            array128[2].sd128 = do2_boxmuller128(ar[2], axy[2], csigma, cmu);
            return p * 2;
        case 2:
            array128[0].sd128 = do2_boxmuller128(ar[0], axy[0], csigma, cmu);
            array128[1].sd128 = do2_boxmuller128(ar[1], axy[1], csigma, cmu);
            return p * 2;
        case 1:
            array128[0].sd128 = do2_boxmuller128(ar[0], axy[0], csigma, cmu);
            return p * 2;
        case 0:
        default:
            return 0;
        }
        _mm256_zeroall();
    }
#endif
#else // don't HAVE_AVX2
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
    void fillArray256_maxint(double *, int32_t *, uint64_t, int32_t)
    {
        throw new std::logic_error("should not be called");
    }
#if 0
    int fillArray256_boxmuller(double *, double *, double, double)
    {
        throw new std::logic_error("should not be called");
    }
#endif
#endif // HAVE_AVX2

    /**
     * This function represents the recursion formula.
     * @param p parameter
     * @param r output
     * @param a a 256-bit part of the internal state array
     * @param b a 256-bit part of the internal state array
     * @param lung a 256-bit part of the internal state array
     */
    inline static void recursion64(w256_t *r,
                                   w256_t *a, w256_t * b, w256_t *lung)
    {
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
        w256_t t;
        lung->u64[0] ^= b->u64[0];
        lung->u64[1] ^= b->u64[1];
        lung->u64[2] ^= b->u64[2];
        lung->u64[3] ^= b->u64[3];
        lung->u64[0] ^= a->u64[0] << DSFMTAVX2_SL1;
        lung->u64[1] ^= a->u64[1] << DSFMTAVX2_SL1;
        lung->u64[2] ^= a->u64[2] << DSFMTAVX2_SL1;
        lung->u64[3] ^= a->u64[3] << DSFMTAVX2_SL1;
        t.u64[0] = lung->u64[0] >> DSFMTAVX2_SR;
        t.u64[1] = lung->u64[1] >> DSFMTAVX2_SR;
        t.u64[2] = lung->u64[2] >> DSFMTAVX2_SR;
        t.u64[3] = lung->u64[3] >> DSFMTAVX2_SR;
        t.u64[0] ^= a->u64[0];
        t.u64[1] ^= a->u64[1];
        t.u64[2] ^= a->u64[2];
        t.u64[3] ^= a->u64[3];
        r->u64[0] = t.u64[0] ^ (lung->u64[0] & mask1.u64[0]);
        r->u64[1] = t.u64[1] ^ (lung->u64[1] & mask1.u64[1]);
        r->u64[2] = t.u64[2] ^ (lung->u64[2] & mask1.u64[2]);
        r->u64[3] = t.u64[3] ^ (lung->u64[3] & mask1.u64[3]);
    }

    inline  void convert64_c0o1(w256_t *w) {
        w->d[0] -= 1.0;
        w->d[1] -= 1.0;
        w->d[2] -= 1.0;
        w->d[3] -= 1.0;
    }

    inline  void convert64_o0c1(w256_t *w) {
        w->d[0] = 2.0 - w->d[0];
        w->d[1] = 2.0 - w->d[1];
        w->d[2] = 2.0 - w->d[2];
        w->d[3] = 2.0 - w->d[3];
    }

    inline  void convert64_o0o1(w256_t *w) {
        w->u64[0] |= 1;
        w->u64[1] |= 1;
        w->u64[2] |= 1;
        w->u64[3] |= 1;
        w->d[0] -= 1.0;
        w->d[1] -= 1.0;
        w->d[2] -= 1.0;
        w->d[3] -= 1.0;
    }

    /**
     * This function fills the internal state array with pseudorandom
     * integers.
     * @param p parameter
     * @param sfmt SFMT internal state
     */
    void fillState64(double * state64)
    {
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t lung = state[2];
        recursion64(&state[0], &state[0],
                    &state[1], &lung);
        recursion64(&state[1], &state[1],
                    &state[0], &lung);
        state[2] = lung;
    }

    void fillArray64_c1o2(double * state64,
                          double * array64, int length)
    {
        DMSG("start fillArray64_c1o2");
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t *lung = &state[2];
        recursion64(&array[0], &state[0], &state[1], lung);
        recursion64(&array[1], &state[1], &array[0], lung);
        for (int i = 2; i < length; i++) {
            recursion64(&array[i], &array[i - 2], &array[i - 1], lung);
        }
        state[0] = array[length - 2];
        state[1] = array[length - 1];
    }

    void fillArray64_c0o1(double * state64,
                          double * array64, int length)
    {
        DMSG("start fillArray64_c0o1");
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t *lung = &state[2];
        recursion64(&array[0], &state[0], &state[1], lung);
        recursion64(&array[1], &state[1], &array[0], lung);
        for (int i = 2; i < length; i++) {
            recursion64(&array[i], &array[i - 2], &array[i - 1], lung);
            convert64_c0o1(&array[i - 2]);
        }
        state[0] = array[length - 2];
        state[1] = array[length - 1];
        convert64_c0o1(&array[length - 2]);
        convert64_c0o1(&array[length - 1]);
    }

    void fillArray64_o0c1(double * state64, double * array64, int length)
    {
        DMSG("start fillArray64_o0c1");
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t *lung = &state[2];
        recursion64(&array[0], &state[0], &state[1], lung);
        recursion64(&array[1], &state[1], &array[0], lung);
        for (int i = 2; i < length; i++) {
            recursion64(&array[i], &array[i - 2], &array[i - 1], lung);
            convert64_o0c1(&array[i - 2]);
        }
        state[0] = array[length - 2];
        state[1] = array[length - 1];
        convert64_o0c1(&array[length - 2]);
        convert64_o0c1(&array[length - 1]);
    }

    void fillArray64_o0o1(double * state64, double * array64, int length)
    {
        DMSG("start fillArray64");
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t *lung = &state[2];
        recursion64(&array[0], &state[0], &state[1], lung);
        recursion64(&array[1], &state[1], &array[0], lung);
        for (int i = 2; i < length; i++) {
            recursion64(&array[i], &array[i - 2], &array[i - 1], lung);
            convert64_o0o1(&array[i - 2]);
        }
        state[0] = array[length - 2];
        state[1] = array[length - 1];
        convert64_o0o1(&array[length - 2]);
        convert64_o0o1(&array[length - 1]);
    }

    /* array_size = 2 */
    inline int32_t do64_maxint(double in, double dmax, int32_t min)
    {
        return static_cast<int32_t>((in - 1.0) * dmax) + min;
    }

    void fillArray64_maxint(double * state, int32_t * array,
                            uint64_t rmax, int32_t min)
    {
        double dmax = static_cast<double>(rmax);
        fillState64(state);
        array[0] = do64_maxint(state[0], dmax, min);
        array[1] = do64_maxint(state[1], dmax, min);
        array[2] = do64_maxint(state[2], dmax, min);
        array[3] = do64_maxint(state[3], dmax, min);
        array[4] = do64_maxint(state[4], dmax, min);
        array[5] = do64_maxint(state[5], dmax, min);
        array[6] = do64_maxint(state[6], dmax, min);
        array[7] = do64_maxint(state[7], dmax, min);
        fillState64(state);
        array[8] = do64_maxint(state[0], dmax, min);
        array[9] = do64_maxint(state[1], dmax, min);
        array[10] = do64_maxint(state[2], dmax, min);
        array[11] = do64_maxint(state[3], dmax, min);
        array[12] = do64_maxint(state[4], dmax, min);
        array[13] = do64_maxint(state[5], dmax, min);
        array[14] = do64_maxint(state[6], dmax, min);
        array[15] = do64_maxint(state[7], dmax, min);
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
        for (int i = 0; i < array_size; i++) {
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
    void period_certification(w256_t * state)
    {
        w256_t tmp;
        uint64_t inner;
        int size = array_size;
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

        int size = array_size;
        psfmt = &state[0].u64[0];
        for (i = 0; i < size * 4; i++) {
            psfmt[i] = (psfmt[i] & low_mask) | high_const;
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

        int size = array_size;
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
        int size = (array_size + 1) * 2 * 4;   /* pulmonary */

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
}

namespace MersenneTwister {
    using namespace std;

    DSFMTAVX607::DSFMTAVX607(int, uint32_t seedValue)
    {
        cf = cpu_feature();
        stateSize = array_size;
        stateSize64 = stateSize * 4;
        size_t alloc_size = (stateSize + 1) * 4 * sizeof(uint64_t);
        state = alignedAlloc<double *>(alloc_size);
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue);
    }

    DSFMTAVX607::DSFMTAVX607(uint32_t seedValue)
    {
        cf = cpu_feature();
        stateSize = array_size;
        stateSize64 = stateSize * 4;
        size_t alloc_size = (stateSize + 1) * 4 * sizeof(uint64_t);
        state = alignedAlloc<double *>(alloc_size);
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue);
    }

    DSFMTAVX607::DSFMTAVX607(uint32_t seedValue[], int length)
    {
        cf = cpu_feature();
        stateSize = array_size;
        stateSize64 = stateSize * 4;
        size_t alloc_size = (stateSize + 1) * 4 * sizeof(uint64_t);
        state = alignedAlloc<double *>(alloc_size);
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue, length);
    }

    DSFMTAVX607::~DSFMTAVX607()
    {
        alignedFree(state);
    }

#if defined(DEBUG)
    void DSFMTAVX607::d_p()
    {
        cout << "debug_print" << endl;
        cout << "array_size = " << dec << array_size << endl;
        cout << "mexp = " << dec << mexp << endl;
        cout << "pos1 = " << dec << pos1 << endl;
        cout << "mask[0] = " << hex << mask.u64[0] << endl;
        cout << "mask[1] = " << hex << mask.u64[1] << endl;
        cout << "mask[2] = " << hex << mask.u64[2] << endl;
        cout << "mask[3] = " << hex << mask.u64[3] << endl;
        w256_t * state256 = reinterpret_cast<w256_t *>(state);
        for (int i = 0; i < stateSize + 1; i++) {
            for (int j = 0; j < 4; j++) {
                cout << hex << setfill('0') << setw(16) << state256[i].u64[j]
                     << " ";
            }
            cout << endl;
        }
    }
#endif

    const string DSFMTAVX607::getIDString()
    {
        return ::getIDString();
    }

    /**
     * This function initializes the internal state array with a 32-bit
     * integer seed.
     *
     * @param sfmt SFMT internal state
     * @param seed a 32-bit integer used as the seed.
     */
    void DSFMTAVX607::seed(uint32_t seedValue)
    {
        w256_t * state256 = reinterpret_cast<w256_t *>(state);
        init(state256, seedValue);
        index = stateSize64;
    }

    /**
     * This function initializes the internal state array,
     * with an array of 32-bit integers used as the seeds
     * @param sfmt SFMT internal state
     * @param init_key the array of 32-bit integers, used as a seed.
     * @param key_length the length of init_key.
     */
    void DSFMTAVX607::seed(uint32_t *init_key, int key_length)
    {
        w256_t * state256 = reinterpret_cast<w256_t *>(state);
        init(state256, init_key, key_length);
        index = stateSize64;
    }

    void DSFMTAVX607::fillState()
    {
        if (cf.avx2) {
            fillState256(state);
        } else {
            fillState64(state);
        }
        index = 0;
    }

    void DSFMTAVX607::fillArray(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generate();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
            DMSG("fillArray step 3");
            fillArray256_c0o1(state, array, length / 4);
        } else {
            DMSG("fillArray step 4");
            fillArray64_c0o1(state, array, length / 4);
        }
    }

    void DSFMTAVX607::fillArrayClose1Open2(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateClose1Open2();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
            DMSG("fillArray step 3");
            fillArray256_c1o2(state, array, length / 4);
        } else {
            DMSG("fillArray step 4");
            fillArray64_c1o2(state, array, length / 4);
        }
    }

    void DSFMTAVX607::fillArrayOpen0Close1(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateOpen0Close1();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
            DMSG("fillArray step 3");
            fillArray256_o0c1(state, array, length / 4);
        } else {
            DMSG("fillArray step 4");
            fillArray64_o0c1(state, array, length / 4);
        }
    }

    void DSFMTAVX607::fillArrayOpen0Open1(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateOpen0Open1();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
            DMSG("fillArray step 3");
            fillArray256_o0o1(state, array, length / 4);
        } else {
            DMSG("fillArray step 4");
            fillArray64_o0o1(state, array, length / 4);
        }
    }

    int DSFMTAVX607::getMersenneExponent()
    {
        return 607;
    }

    /*
     * array の要素数はstateの要素数の2倍
     */
    void DSFMTAVX607::fillArrayMaxInt(int32_t * array,
                                        uint64_t rmax, int32_t min)
    {
#if 0
        fillArray64_maxint(state, array, rmax, min);
#else
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
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
    int DSFMTAVX607::fillArrayNormalDist(double * array,
                                           double mu, double sigma)
    {

        int align = reinterpret_cast<uintptr_t>(array) % 32;
        //if ((align == 0) && cf.avx2) {
            //return fillArray256_boxmuller(state, array, mu, sigma);
        //} else
        if ((align % 16 == 0) && cf.sse2) {
            return fillArray128_boxmuller(state, array, mu, sigma);
        } else {
            return fillArray64_boxmuller(state, array, mu, sigma);
        }
    }

    bool DSFMTAVX607::selfTest()
    {
        DMSG("selfTest Start");
        DMSG("fillArray_maxInt test Start");
        if (cf.avx2) {
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
