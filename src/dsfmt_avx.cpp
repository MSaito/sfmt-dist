#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdint.h>
#include "sfmt-dist.h"
#if HAVE_STRING_H
#include <string.h>
#endif
#include "dsfmt_avx.h"
#include <sfmt-dist/cpu_feature.h>
#include <sfmt-dist/aligned_alloc.h>
#include "w256.h"

#if HAVE_ZMMINTRIN_H
#include <zmmintrin.h>
#endif

#if HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif

#include "debug.h"

namespace {
    using namespace MersenneTwister;
    using namespace MersenneTwister::dSFMT_AVX;
    using namespace std;


#define DSFMTAVX_SL1 19
#define DSFMTAVX_SR 12

    const uint64_t low_mask = UINT64_C(0x000FFFFFFFFFFFFF);
    const uint64_t high_const = UINT64_C(0x3FF0000000000000);

    params parameter_array[] = {
        {607,
         2,
         1,
         {{UINT64_C(0x000f7fdedfdaf6f7),
           UINT64_C(0x000ff71b5676ff7f),
           UINT64_C(0x000fb66b7a1ef925),
           UINT64_C(0x000bfaffffff7faf)}},
         {{UINT64_C(0x46f5e1807e0eebd9),
           UINT64_C(0x206a072aabbcf1ca),
           UINT64_C(0xca0610e8704e7f0e),
           UINT64_C(0xba564f522afc7696)}},
         {{UINT64_C(0x0000000000000001),
           UINT64_C(0x0000000000000000),
           UINT64_C(0x4000000000000000),
           UINT64_C(0xb9f61e13e9f75e26)}}
        },
        {1279,
         5,
         3,
         {{UINT64_C(0x000ff9ff37fefbbf),
           UINT64_C(0x0007bebd2cebad7c),
           UINT64_C(0x000bfffbcfaddfce),
           UINT64_C(0x0005affefd978fff)}},
         {{UINT64_C(0xfbc5b6c65831cf64),
           UINT64_C(0x328ec06650089455),
           UINT64_C(0xa6ea0cee0217b775),
           UINT64_C(0x119ade137f700f73)}},
         {{UINT64_C(0x0000000000000001),
           UINT64_C(0x0000000000000000),
           UINT64_C(0x0000000000000000),
           UINT64_C(0xb6c4200000000000)}}
        },
        {2281,
         10,
         5,
         {{UINT64_C(0x00053fdfdbdecff4),
           UINT64_C(0x000fd5edfbdd7fb7),
           UINT64_C(0x000ffcbfb797f6f5),
           UINT64_C(0x000f69effd89efac)}},
         {{UINT64_C(0x7aee281d3142542a),
           UINT64_C(0x176a4fb0881dafba),
           UINT64_C(0x13e626916da50640),
           UINT64_C(0x65b20b60bd5c0674)}},
         {{UINT64_C(0x0000000000000001),
           UINT64_C(0x0000000000000000),
           UINT64_C(0x0000000000000000),
           UINT64_C(0xdbf3c390a41d2200)}}
        },
        {4253,
         20,
         9,
         {{UINT64_C(0x000a9feff9ebff9f),
           UINT64_C(0x000efeffbbf3777f),
           UINT64_C(0x0001bfffdbfdf7bf),
           UINT64_C(0x000c7e3fd779bded)}},
         {{UINT64_C(0xa0cfe3b5ab1db41d),
           UINT64_C(0xb34f9a51d50878c4),
           UINT64_C(0xabebbce7898197fe),
           UINT64_C(0x508ddb4ba74644bf)}},
         {{UINT64_C(0x0000000000000001),
           UINT64_C(0x2ada8a6cc0000000),
           UINT64_C(0xf566075647a7b032),
           UINT64_C(0x4a8f059463e81559)}}
        },
        {19937,
         95,
         47,
         {{UINT64_C(0x000f7eefaefbd7e9),
           UINT64_C(0x000cd7fe2ffcfcc3),
           UINT64_C(0x000ff2fdf7fab37f),
           UINT64_C(0x000cffffd6adff3c)}},
         {{UINT64_C(0x42e415394eb76145),
           UINT64_C(0xb1a401c199de27fd),
           UINT64_C(0x379b5a55f2680707),
           UINT64_C(0x1788815e1a4a4cdd)}},
         {{UINT64_C(0x0000000001000001),
           UINT64_C(0x0000000000000000),
           UINT64_C(0x7c32000000000000),
           UINT64_C(0x4f259872bc33ab2d)}}
        },
        {-1,
         0,
         0,
         {{UINT64_C(0),
           UINT64_C(0),
           UINT64_C(0),
           UINT64_C(0)}},
         {{UINT64_C(0),
           UINT64_C(0),
           UINT64_C(0),
           UINT64_C(0)}},
         {{UINT64_C(0),
           UINT64_C(0),
           UINT64_C(0),
           UINT64_C(0)}}
        }
    };

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
    inline  void recursion256(const params& p,
                              __m256i * r, __m256i * a,
                              __m256i * b, __m256i * u)
    {
        __m256i v, w, x, y, z;
        x = *a;
        z = _mm256_slli_epi64(x, DSFMTAVX_SL1);
        y = _mm256_permutevar8x32_epi32(*u, perm.si256);
        z = _mm256_xor_si256(z, *b);
        y = _mm256_xor_si256(y, z);

        v = _mm256_srli_epi64(y, DSFMTAVX_SR);
        w = _mm256_and_si256(y, p.mask.si256);
        v = _mm256_xor_si256(v, x);
        v = _mm256_xor_si256(v, w);
        *r = v;
        *u = y;
    }

    /**
     * This function fills the internal state array with pseudorandom
     * integers.
     * @param p parameter
     * @param state SFMT internal state
     */
    void fillState256(const params& p, double * state)
    {
        int i = 0;
        __m256i lung;

        int size = p.array_size;
        int pos1 = p.pos1;
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        lung = pstate[size].si256;
        for (; i < size - pos1; i++) {
            recursion256(p, &pstate[i].si256, &pstate[i].si256,
                         &pstate[i + pos1].si256, &lung);
        }
        for (; i < size; i++) {
            recursion256(p, &pstate[i].si256, &pstate[i].si256,
                         &pstate[i + pos1 - size].si256, &lung);
        }
        pstate[size].si256 = lung;
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
    void fillArray256_c1o2(const params& p, double * state,
                           double * array64, int length)
    {
        DMSG("fillArray256_c1o2 step 1");
        int i, j;
        int size = p.array_size;
        int pos1 = p.pos1;
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        __m256i lung = pstate[size].si256;
        DMSG("fillArray256_c1o2 step 2");
        for (i = 0; i < size - pos1; i++) {
            recursion256(p, &array[i].si256, &pstate[i].si256,
                         &pstate[i + pos1].si256, &lung);
        }
        DMSG("fillArray256_c1o2 step 3");
        for (; i < size; i++) {
            recursion256(p, &array[i].si256, &pstate[i].si256,
                         &array[i + pos1 - size].si256, &lung);
        }
        DMSG("fillArray256 step 6");
        for (; i < length; i++) {
            recursion256(p, &array[i].si256, &array[i - p.array_size].si256,
                         &array[i + pos1 - size].si256, &lung);
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            pstate[j++].si256 = array[i].si256;
        }
        pstate[size].si256 = lung;
        _mm256_zeroall();
    }

    inline  void convert256_c0o1(w256_t *w) {
        w->sd256 = _mm256_add_pd(w->sd256, m_one.sd256);
    }

    inline  void convert256_o0c1(w256_t *w) {
        w->sd256 = _mm256_sub_pd(two.sd256, w->sd256);
    }

    inline  void convert256_o0o1(w256_t *w) {
        w->si256 = _mm256_or_si256(w->si256, one.si256);
        w->sd256 = _mm256_add_pd(w->sd256, m_one.sd256);
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param p parameter
     * @param state SFMT internal state.
     * @param array64 an 256-bit array to be filled by pseudorandom numbers.
     * @param length number of 256-bit pseudorandom numbers to be generated.
     */
    void fillArray256_c0o1(const params& p, double * state,
                           double * array64, int length)
    {
        DMSG("fillArray256_c0o1 step 1");
        int i, j;
        int size = p.array_size;
        int pos1 = p.pos1;
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        __m256i lung = pstate[size].si256;
        DMSG("fillArray256_c0o1 step 2");
        for (i = 0; i < size - pos1; i++) {
            recursion256(p, &array[i].si256, &pstate[i].si256,
                         &pstate[i + pos1].si256, &lung);
        }
        DMSG("fillArray256_c0o1 step 3");
        for (; i < size; i++) {
            recursion256(p, &array[i].si256, &pstate[i].si256,
                         &array[i + pos1 - size].si256, &lung);
        }
        DMSG("fillArray256_c0o1 step 6");
        for (; i < length; i++) {
            recursion256(p, &array[i].si256, &array[i - size].si256,
                         &array[i + pos1 - size].si256, &lung);
            convert256_c0o1(&array[i - size]);
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            pstate[j++].si256 = array[i].si256;
            convert256_c0o1(&array[i]);
        }
        pstate[size].si256 = lung;
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
    void fillArray256_o0c1(const params& p, double * state,
                           double * array64, int length)
    {
        DMSG("fillArray256_o0c1 step 1");
        int i, j;
        int size = p.array_size;
        int pos1 = p.pos1;
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        __m256i lung = pstate[size].si256;
        DMSG("fillArray256_o0c1 step 2");
        for (i = 0; i < size - pos1; i++) {
            recursion256(p, &array[i].si256, &pstate[i].si256,
                         &pstate[i + pos1].si256, &lung);
        }
        DMSG("fillArray256_o0c1 step 3");
        for (; i < size; i++) {
            recursion256(p, &array[i].si256, &pstate[i].si256,
                         &array[i + pos1 - size].si256, &lung);
        }
        DMSG("fillArray256_o0c1 step 6");
        for (; i < length; i++) {
            recursion256(p, &array[i].si256, &array[i - size].si256,
                         &array[i + pos1 - size].si256, &lung);
            convert256_o0c1(&array[i - size]);
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            pstate[j++].si256 = array[i].si256;
            convert256_o0c1(&array[i]);
        }
        pstate[size].si256 = lung;
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
    void fillArray256_o0o1(const params& p, double * state,
                           double * array64, int length)
    {
        DMSG("fillArray256 step 1");
        int i, j;
        int size = p.array_size;
        int pos1 = p.pos1;
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        __m256i lung = pstate[size].si256;
        DMSG("fillArray256 step 2");
        for (i = 0; i < size - pos1; i++) {
            recursion256(p, &array[i].si256, &pstate[i].si256,
                         &pstate[i + pos1].si256, &lung);
        }
        DMSG("fillArray256 step 3");
        for (; i < size; i++) {
            recursion256(p, &array[i].si256, &pstate[i].si256,
                         &array[i + pos1 - size].si256, &lung);
        }
        DMSG("fillArray256 step 6");
        for (; i < length; i++) {
            recursion256(p, &array[i].si256, &array[i - size].si256,
                         &array[i + pos1 - size].si256, &lung);
            convert256_o0o1(&array[i - size]);
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            pstate[j++].si256 = array[i].si256;
            convert256_o0o1(&array[i]);
        }
        pstate[size].si256 = lung;
        _mm256_zeroall();
    }
#else // don't HAVE_AVX2
    void fillState256(const params&, double *)
    {
        throw new std::logic_error("should not be called");
    }

    void fillArray256_c1o2(const params&, double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray256_c0o1(const params&, double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray256_o0c1(const params&, double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray256_o0o1(const params&, double *, double *, int)
    {
        throw new std::logic_error("should not be called");
    }
#endif // HAVE_AVX2

    /**
     * This function represents the recursion formula.
     * @param p parameter
     * @param r output
     * @param a a 256-bit part of the internal state array
     * @param b a 256-bit part of the internal state array
     * @param lung a 256-bit part of the internal state array
     */
    inline  void recursion64(const params& p, w256_t *r,
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
        lung->u64[0] ^= b->u64[0];
        lung->u64[1] ^= b->u64[1];
        lung->u64[2] ^= b->u64[2];
        lung->u64[3] ^= b->u64[3];
        lung->u64[0] ^= a->u64[0] << DSFMTAVX_SL1;
        lung->u64[1] ^= a->u64[1] << DSFMTAVX_SL1;
        lung->u64[2] ^= a->u64[2] << DSFMTAVX_SL1;
        lung->u64[3] ^= a->u64[3] << DSFMTAVX_SL1;
        w256_t t;
        t.u64[0] = lung->u64[0] >> DSFMTAVX_SR;
        t.u64[1] = lung->u64[1] >> DSFMTAVX_SR;
        t.u64[2] = lung->u64[2] >> DSFMTAVX_SR;
        t.u64[3] = lung->u64[3] >> DSFMTAVX_SR;
        t.u64[0] ^= a->u64[0];
        t.u64[1] ^= a->u64[1];
        t.u64[2] ^= a->u64[2];
        t.u64[3] ^= a->u64[3];
        r->u64[0] = t.u64[0] ^ (lung->u64[0] & p.mask.u64[0]);
        r->u64[1] = t.u64[1] ^ (lung->u64[1] & p.mask.u64[1]);
        r->u64[2] = t.u64[2] ^ (lung->u64[2] & p.mask.u64[2]);
        r->u64[3] = t.u64[3] ^ (lung->u64[3] & p.mask.u64[3]);
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
    void fillState64(const params& p, double * state64)
    {
        int i;
        int size = p.array_size;
        int pos1 = p.pos1;
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t lung = state[size];
        for (i = 0; i < size - pos1; i++) {
            recursion64(p, &state[i], &state[i],
                        &state[i + pos1], &lung);
        }
        for (; i < size; i++) {
            recursion64(p, &state[i], &state[i],
                        &state[i + pos1 - size], &lung);
        }
        state[size] = lung;
    }

    void fillArray64_c1o2(const params& p, double * state64,
                          double * array64, int length)
    {
        int i, j;
        DMSG("start fillArray64_c1o2");

        int size = p.array_size;
        int pos1 = p.pos1;
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t lung = state[size];
        for (i = 0; i < size - pos1; i++) {
            recursion64(p, &array[i], &state[i], &state[i + pos1], &lung);
        }
        for (; i < size; i++) {
            recursion64(p, &array[i], &state[i],
                        &array[i + pos1 - size], &lung);
        }
        for (; i < length; i++) {
            recursion64(p, &array[i], &array[i - size],
                        &array[i + pos1 - size], &lung);
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            state[j++] = array[i];
        }
        state[size] = lung;
    }

    void fillArray64_c0o1(const params& p, double * state64,
                          double * array64, int length)
    {
        int i, j;
        DMSG("start fillArray64_c0o1");

        int size = p.array_size;
        int pos1 = p.pos1;
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t lung = state[size];
        for (i = 0; i < size - pos1; i++) {
            recursion64(p, &array[i], &state[i], &state[i + pos1], &lung);
        }
        for (; i < size; i++) {
            recursion64(p, &array[i], &state[i],
                        &array[i + pos1 - size], &lung);
        }
        for (; i < length; i++) {
            recursion64(p, &array[i], &array[i - size],
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

    void fillArray64_o0c1(const params& p, double * state64,
                          double * array64, int length)
    {
        int i, j;
        DMSG("start fillArray64_o0c1");

        int size = p.array_size;
        int pos1 = p.pos1;
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t lung = state[size];
        for (i = 0; i < size - pos1; i++) {
            recursion64(p, &array[i], &state[i], &state[i + pos1], &lung);
        }
        for (; i < size; i++) {
            recursion64(p, &array[i], &state[i],
                        &array[i + pos1 - size], &lung);
        }
        for (; i < length; i++) {
            recursion64(p, &array[i], &array[i - size],
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

    void fillArray64_o0o1(const params& p, double * state64,
                          double * array64, int length)
    {
        int i, j;
        DMSG("start fillArray64");

        int size = p.array_size;
        int pos1 = p.pos1;
        w256_t * state = reinterpret_cast<w256_t *>(state64);
        w256_t * array = reinterpret_cast<w256_t *>(array64);
        w256_t lung = state[size];
        for (i = 0; i < size - pos1; i++) {
            recursion64(p, &array[i], &state[i], &state[i + pos1], &lung);
        }
        for (; i < size; i++) {
            recursion64(p, &array[i], &state[i],
                        &array[i + pos1 - size], &lung);
        }
        for (; i < length; i++) {
            recursion64(p, &array[i], &array[i - size],
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
}

namespace MersenneTwister {
    namespace dSFMT_AVX {
        using namespace std;

        params * search_params(int mexp)
        {
            for (params * p = &parameter_array[0];;p++) {
                if (p->mexp < 0 || p->mexp > mexp) {
                    return NULL;
                }
                if (p->mexp == mexp) {
                    return p;
                }
            }
            return NULL;
        }

        /**
         * This function initializes the internal state array to fit the IEEE
         * 754 format.
         * @param dsfmt dsfmt state vector.
         */
        void initial_mask(w256_t * state, int state_size)
        {
            int size = state_size;
            uint64_t * psfmt = &state[0].u64[0];
            for (int i = 0; i < size * 4; i++) {
                psfmt[i] = (psfmt[i] & low_mask) | high_const;
            }
        }

        string get_id_string(const params& params)
        {
            stringstream ss;
            char delim = ':';
            ss << "dSFMTAVX-" << dec << params.mexp
               << ":" << dec << params.pos1 << "-" << DSFMTAVX_SL1;
            for (int i = 0; i < 4; i++) {
                ss << delim << setfill('0') << setw(13) << hex
                   << params.mask.u64[i];
                delim = '-';
            }
            string s;
            ss >> s;
            return s;
        }

        /**
         * This function certificate the period of 2^{SFMT_MEXP}-1.
         * @param dsfmt dsfmt state vector.
         */
        void period_certification(const params& p, w256_t * state)
        {
            w256_t tmp;
            uint64_t inner;
            int size = p.array_size;
            for (int i = 0; i < 4; i++) {
                tmp.u64[i] = state[size].u64[i] ^ p.fix.u64[i];
            }
            inner = 0;
            for (int i = 0; i < 4; i++) {
                inner ^= tmp.u64[i] & p.pcv.u64[i];
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
            if ((p.pcv.u64[0] & 1) == 1) {
                state[size].u64[0] ^= 1;
                return;
            }
            int i;
            int j;
            uint64_t work;
            for (i = 0; i < 4; i++) {
                work = 1;
                for (j = 0; j < 64; j++) {
                    if ((work & p.pcv.u64[i]) != 0) {
                        state[size].u64[i] ^= work;
                        return;
                    }
                    work = work << 1;
                }
            }
        }

        void fill_state(const cpu_feature_t& cf,
                        const params& params,
                        double * state)
        {
            if (cf.avx2) {
                fillState256(params, state);
            } else {
                fillState64(params, state);
            }
        }

        void fill_array(const cpu_feature_t&cf,
                        const params& params,
                        double * state, double * array, int length)
        {
            int align = reinterpret_cast<uintptr_t>(array) % 32;
            if ((align == 0) && cf.avx2) {
                DMSG("fillArray step 3");
                fillArray256_c0o1(params, state, array, length / 4);
            } else {
                DMSG("fillArray step 4");
                fillArray64_c0o1(params, state, array, length / 4);
            }
        }

        void fill_arrayc1o2(const cpu_feature_t&cf,
                            const params& params,
                            double * state,
                            double * array,
                            int length)
        {
            int align = reinterpret_cast<uintptr_t>(array) % 32;
            if ((align == 0) && cf.avx2) {
                DMSG("fillArray step 3");
                fillArray256_c1o2(params, state, array, length / 4);
            } else {
                DMSG("fillArray step 4");
                fillArray64_c1o2(params, state, array, length / 4);
            }
        }

        void fill_arrayo0c1(const cpu_feature_t&cf,
                            const params& params,
                            double * state,
                            double * array,
                            int length)
        {
            int align = reinterpret_cast<uintptr_t>(array) % 32;
            if ((align == 0) && cf.avx2) {
                DMSG("fillArray step 3");
                fillArray256_o0c1(params, state, array, length / 4);
            } else {
                DMSG("fillArray step 4");
                fillArray64_o0c1(params, state, array, length / 4);
            }
        }

        void fill_arrayo0o1(const cpu_feature_t&cf,
                            const params& params,
                            double * state,
                            double * array,
                            int length)
        {
            int align = reinterpret_cast<uintptr_t>(array) % 32;
            if ((align == 0) && cf.avx2) {
                DMSG("fillArray step 3");
                fillArray256_o0o1(params, state, array, length / 4);
            } else {
                DMSG("fillArray step 4");
                fillArray64_o0o1(params, state, array, length / 4);
            }
        }
    }
}
