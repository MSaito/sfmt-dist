#include "sfmt-dist.h"
#include <sfmt-dist/sfmt19937.h>
#include <sfmt-dist/cpu_feature.h>
#include <sfmt-dist/aligned_alloc.h>
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

namespace {
#define SFMT_N          156
#define SFMT_POS1       122
#define SFMT_SL1        18
#define SFMT_SL2        1
#define SFMT_SR1        11
#define SFMT_SR2        1
#define SFMT_MSK1       0xdfffffefU
#define SFMT_MSK2       0xddfecb7fU
#define SFMT_MSK3       0xbffaffffU
#define SFMT_MSK4       0xbffffff6U
#define SFMT_PARITY1    0x00000001U
#define SFMT_PARITY2    0x00000000U
#define SFMT_PARITY3    0x00000000U
#define SFMT_PARITY4    0x13c9e684U

    const int size = SFMT_N;
    const int pos1 = SFMT_POS1;
    //const int sl1 = SFMT_SL1;
    //const int sl2 = SFMT_SL2;
    //const int sr1 = SFMT_SR1;
    //const int sr2 = SFMT_SR2;
    //const uint32_t msk1 = SFMT_MSK1;
    //const uint32_t msk2 = SFMT_MSK2;
    //const uint32_t msk3 = SFMT_MSK3;
    //const uint32_t msk4 = SFMT_MSK4;
    //const uint32_t parity1 = SFMT_PARITY1;
    //const uint32_t parity2 = SFMT_PARITY2;
    //const uint32_t parity3 = SFMT_PARITY3;
    //const uint32_t parity4 = SFMT_PARITY4;
    const char * idString
    = "SFMT-19937:122-18-1-11-1:dfffffef-ddfecb7f-bffaffff-bffffff6";

    const int size32 = SFMT_N * 4;

#ifdef ONLY64
    inline static int idxof(int i) {
        return i ^ 1;
    }
#else
    inline static int idxof(int i) {
        return i;
    }
#endif

#if HAVE_AVX2
    union w256_t {
        uint32_t u[8];
        uint64_t u64[4];
        __m256i si256;
    };

    const w256_t msk256 = {{SFMT_MSK1, SFMT_MSK2, SFMT_MSK3, SFMT_MSK4,
                            SFMT_MSK1, SFMT_MSK2, SFMT_MSK3, SFMT_MSK4}};

    /**
     * @author SeizhLab
     * @note: https://github.com/seizh/sfmt_mod
     */
    inline static __m256i recursion256(__m256i a, __m256i b, __m256i c)
    {
        __m256i x, y, z;

        x = _mm256_slli_si256(a, SFMT_SL2);
        x = _mm256_xor_si256(x, a);
        y = _mm256_srli_epi32(b, SFMT_SR1);
        y = _mm256_and_si256(y, msk256.si256);
        x = _mm256_xor_si256(x, y);
        z = _mm256_srli_si256(c, SFMT_SR2);
        x = _mm256_xor_si256(x, z);

        /* assume SFMT_SL1 >= 16 */
        z = _mm256_permute2f128_si256(c, x, 0x21);      /* [c.upper, x.lower] */
        z = _mm256_slli_epi32(z, SFMT_SL1);
        x = _mm256_xor_si256(x, z);

        return x;
    }

    /**
     * This function fills the internal state array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state
     */
    void fillState256(uint32_t * state)
    {
        DMSG("fillState256 start\n");
        int i;
        __m256i r;
        __m256i x;
        __m256i y;
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        int ppos = pos1 / 2;
        int sfmt_n = SFMT_N / 2;
#if defined(DEBUG) && 0
#include <inttypes.h>
        for (int i = 0; i < 8; i++) {
            printf("pstate[0].u[%d] = %" PRIu32 "\n", i, pstate[0].u[i]);
        }
        for (int i = 0; i < 8; i++) {
            printf("pstate[pos1].u[%d] = %" PRIu32 "\n", i, pstate[ppos].u[i]);
        }
        for (int i = 0; i < 8; i++) {
            printf("pstate[SFMT_N - 2].u[%d] = %" PRIu32 "\n",
                   i, pstate[sfmt_n -1].u[i]);
        }
        r = _mm256_load_si256(&pstate[sfmt_n - 1].si256);
        x = _mm256_load_si256(&pstate[0].si256);
        y = _mm256_load_si256(&pstate[ppos].si256);
        w256_t w;
        w.si256 = r;
        for (int i = 0; i < 8; i++) {
            printf("r.u[%d] = %" PRIu32 "\n", i, w.u[i]);
        }
        w.si256 = x;
        for (int i = 0; i < 8; i++) {
            printf("x.u[%d] = %" PRIu32 "\n", i, w.u[i]);
        }
        w.si256 = y;
        for (int i = 0; i < 8; i++) {
            printf("y.u[%d] = %" PRIu32 "\n", i, w.u[i]);
        }
        r = recursion256(x, y, r);
        w.si256 = r;
        for (int i = 0; i < 8; i++) {
            printf("r.u[%d] = %" PRIu32 "\n", i, w.u[i]);
        }
#endif

        r = _mm256_load_si256(&pstate[sfmt_n - 1].si256);
        // pos1 is an even number
        for (i = 0; i < sfmt_n - ppos; i++) {
            x = _mm256_load_si256(&pstate[i].si256);
            y = _mm256_load_si256(&pstate[i + ppos].si256);
            r = recursion256(x, y, r);
            _mm256_store_si256(&pstate[i].si256, r);
        }
        for (; i < sfmt_n; i++) {
            x = _mm256_load_si256(&pstate[i].si256);
            y = _mm256_load_si256(&pstate[i + ppos - sfmt_n].si256);
            r = recursion256(x, y, r);
            _mm256_store_si256(&pstate[i].si256, r);
        }
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state.
     * @param array an 256-bit array to be filled by pseudorandom numbers.
     * @param size number of 256-bit pseudorandom numbers to be generated.
     */
    void fillArray256(uint32_t * state, uint32_t * array32, int length)
    {
        DMSG("fillArray256 step 1");
        int i;
        __m256i r;
        __m256i x;
        __m256i y;
        int psize = size / 2;
        length = length / 2;
        // pos1 is an even number
        int ppos = pos1 / 2;
        w256_t * pstate = reinterpret_cast<w256_t *>(state);
        w256_t * array = reinterpret_cast<w256_t *>(array32);
#if defined(DEBUG) && 0
#include <inttypes.h>
        for (int i = 0; i < 8; i++) {
            printf("pstate[0].u[%d] = %" PRIu32 "\n", i, pstate[0].u[i]);
        }
        for (int i = 0; i < 8; i++) {
            printf("pstate[pos1].u[%d] = %" PRIu32 "\n", i, pstate[ppos].u[i]);
        }
        for (int i = 0; i < 8; i++) {
            printf("pstate[SFMT_N - 2].u[%d] = %" PRIu32 "\n",
                   i, pstate[psize -1].u[i]);
        }
        r = _mm256_load_si256(&pstate[psize - 1].si256);
        x = _mm256_load_si256(&pstate[0].si256);
        y = _mm256_load_si256(&pstate[ppos].si256);
        w256_t w;
        w.si256 = r;
        for (int i = 0; i < 8; i++) {
            printf("r.u[%d] = %" PRIu32 "\n", i, w.u[i]);
        }
        w.si256 = x;
        for (int i = 0; i < 8; i++) {
            printf("x.u[%d] = %" PRIu32 "\n", i, w.u[i]);
        }
        w.si256 = y;
        for (int i = 0; i < 8; i++) {
            printf("y.u[%d] = %" PRIu32 "\n", i, w.u[i]);
        }
        r = recursion256(x, y, r);
        w.si256 = r;
        for (int i = 0; i < 8; i++) {
            printf("r.u[%d] = %" PRIu32 "\n", i, w.u[i]);
        }
#endif
        r = _mm256_load_si256(&pstate[psize - 1].si256);
        DMSG("fillArray256 step 2");
        for (i = 0; i < psize - ppos; i++) {
            x = _mm256_load_si256(&pstate[i].si256);
            y = _mm256_load_si256(&pstate[i + ppos].si256);
            r = recursion256(x, y, r);
            _mm256_store_si256(&array[i].si256, r);
        }
        DMSG("fillArray256 step 3");
        for (; i < psize; i++) {
            x = _mm256_load_si256(&pstate[i].si256);
            y = _mm256_load_si256(&array[i + ppos - psize].si256);
            r = recursion256(x, y, r);
            _mm256_store_si256(&array[i].si256, r);
        }
        DMSG("fillArray256 step 6");
        for (; i < length; i++) {
            x = _mm256_load_si256(&array[i - psize].si256);
            y = _mm256_load_si256(&array[i + ppos - psize].si256);
            r = recursion256(x, y, r);
            _mm256_store_si256(&array[i].si256, r);
        }
        int j = 0;
        for (i = length - psize; i < length; i++) {
            x = _mm256_load_si256(&array[i].si256);
            _mm256_store_si256(&pstate[j].si256, x);
            j++;
        }
    }
#if 0
    void fillArrayMaxMaskAVX2(uint32_t * state, uint32_t * array,
                              uint32_t mask, int shift)
    {
        __m256i mask256 = _mm256_set1_epi32(mask);
        __m128i shift128 = _mm_set1_epi32(shift);
        w256_t * state256 = reinterpret_cast<w256_t *>(state);
        w256_t * array256 = reinterpret_cast<w256_t *>(array);
        for (int i = 0; i < size / 2; i++) {
            __m256i x = _mm256_srl_epi32(state256[i].si256, shift128);
            array256[i].si256 = _mm256_and_si256(x, mask256);
        }
    }
#endif
#else
    void fillState256(uint32_t *)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray256(uint32_t *, uint32_t *, int)
    {
        throw new std::logic_error("should not be called");
    }
#if 0
    void fillArrayMaxMaskAVX2(uint32_t *, uint32_t *, uint32_t, int)
    {
        throw new std::logic_error("should not be called");
    }
#endif
#endif

#if HAVE_SSE2 || HAVE_SSSE3 || HAVE_SSE4_2 || HAVE_AVX2
#include <emmintrin.h>
    union w128_t {
        uint32_t u[4];
        uint64_t u64[2];
        __m128i si128;
    };

    const w128_t msk128 = {{SFMT_MSK1, SFMT_MSK2, SFMT_MSK3, SFMT_MSK4}};

    /**
     * This function represents the recursion formula.
     * @param a a 128-bit part of the interal state array
     * @param b a 128-bit part of the interal state array
     * @param c a 128-bit part of the interal state array
     * @param d a 128-bit part of the interal state array
     * @return result
     */
    inline static __m128i recursion128(__m128i a, __m128i b,
                                       __m128i c, __m128i d)
    {
        __m128i v, x, y, z;

        y = _mm_srli_epi32(b, SFMT_SR1);
        z = _mm_srli_si128(c, SFMT_SR2);
        v = _mm_slli_epi32(d, SFMT_SL1);
        z = _mm_xor_si128(z, a);
        z = _mm_xor_si128(z, v);
        x = _mm_slli_si128(a, SFMT_SL2);
        y = _mm_and_si128(y, msk128.si128);
        z = _mm_xor_si128(z, x);
        z = _mm_xor_si128(z, y);
        return z;
    }

    /**
     * This function fills the internal state array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state
     */
    void fillState128(uint32_t * state) {
        int i;
        __m128i r1, r2;
        w128_t * pstate = reinterpret_cast<w128_t *>(state);
#if defined(DEBUG) && 0
#include <inttypes.h>
        for (int i = 0; i < 4; i++) {
            printf("pstate[0].u[%d] = %" PRIu32 "\n", i, pstate[0].u[i]);
        }
        for (int i = 0; i < 4; i++) {
            printf("pstate[pos1].u[%d] = %" PRIu32 "\n", i, pstate[pos1].u[i]);
        }
        for (int i = 0; i < 4; i++) {
            printf("pstate[SFMT_N - 1].u[%d] = %" PRIu32 "\n",
                   i, pstate[SFMT_N -1].u[i]);
        }
        for (int i = 0; i < 4; i++) {
            printf("pstate[SFMT_N - 2].u[%d] = %" PRIu32 "\n",
                   i, pstate[SFMT_N -2].u[i]);
        }
        r1 =  recursion128(pstate[0].si128, pstate[pos1].si128,
                           pstate[SFMT_N -1].si128, pstate[SFMT_N -1].si128);
        w128_t w;
        w.si128 = r1;
        for (int i = 0; i < 4; i++) {
            printf("r.u[%d] = %" PRIu32 "\n", i, w.u[i]);
        }
#endif
        r1 = pstate[SFMT_N - 2].si128;
        r2 = pstate[SFMT_N - 1].si128;
        for (i = 0; i < SFMT_N - pos1; i++) {
            pstate[i].si128 = recursion128(pstate[i].si128,
                                        pstate[i + pos1].si128, r1, r2);
            r1 = r2;
            r2 = pstate[i].si128;
        }
        for (; i < SFMT_N; i++) {
            pstate[i].si128 = recursion128(pstate[i].si128,
                                        pstate[i + pos1 - size].si128,
                                        r1, r2);
            r1 = r2;
            r2 = pstate[i].si128;
        }
    }

    /**
     * This function fills the user-specified array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state.
     * @param array an 128-bit array to be filled by pseudorandom numbers.
     * @param size number of 128-bit pseudorandom numbers to be generated.
     */
    static void fillArray128(uint32_t * state, uint32_t * array32, int length)
    {
        DMSG("fillArray128 step 1");
        int i, j;
        __m128i r1, r2;
        w128_t * pstate = reinterpret_cast<w128_t *>(state);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        r1 = pstate[size - 2].si128;
        r2 = pstate[size - 1].si128;
        DMSG("fillArray128 step 2");
        for (i = 0; i < size - pos1; i++) {
            array[i].si128 = recursion128(pstate[i].si128,
                                       pstate[i + pos1].si128, r1, r2);
            r1 = r2;
            r2 = array[i].si128;
        }
        DMSG("fillArray128 step 3");
        for (; i < size; i++) {
            array[i].si128 = recursion128(pstate[i].si128,
                                       array[i + pos1 - size].si128,
                                       r1, r2);
            r1 = r2;
            r2 = array[i].si128;
        }

        DMSG("fillArray128 step 6");
        for (; i < length; i++) {
            array[i].si128 = recursion128(array[i - size].si128,
                                       array[i + pos1 - size].si128, r1, r2);
            r1 = r2;
            r2 = array[i].si128;
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            pstate[j++] = array[i];
        }
    }
#if 0
    void fillArrayMaxMaskSSE2(uint32_t * state, uint32_t * array,
                              uint32_t mask, int shift)
    {
        __m128i mask128 = _mm_set1_epi32(mask);
        __m128i shift128 = _mm_set1_epi32(shift);
        w128_t * state128 = reinterpret_cast<w128_t *>(state);
        w128_t * array128 = reinterpret_cast<w128_t *>(array);
        for (int i = 0; i < size; i++) {
            __m128i x = _mm_srl_epi32(state128[i].si128, shift128);
            array128[i].si128 = _mm_and_si128(x, mask128);
        }
    }

    int fillArrayMaxMaskUint8SSE2(uint32_t * state, uint8_t * array,
                                  uint8_t max, uint32_t mask, int shift)
    {
        union w128xx_t {
            __m128i si128;
            uint8_t u8[16];
        } x;
        int p = 0;
        __m128i mask128 = _mm_set1_epi32(mask);
        __m128i shift128 = _mm_set1_epi32(shift);
        w128_t * state128 = reinterpret_cast<w128_t *>(state);
        //w128_t * array128 = reinterpret_cast<w128_t *>(array);
#define setu8(idx) array[p] = x.u8[idx]; p += (x.u8[idx] <= max) & 1
        for (int i = 0; i < size; i++) {
            x.si128 = _mm_srl_epi32(state128[i].si128, shift128);
            x.si128 = _mm_and_si128(x.si128, mask128);
//            array128[i].si128 = _mm_and_si128(x, mask128);
            setu8(0);
            setu8(1);
            setu8(2);
            setu8(3);
            setu8(4);
            setu8(5);
            setu8(6);
            setu8(7);
            setu8(8);
            setu8(9);
            setu8(10);
            setu8(11);
            setu8(12);
            setu8(13);
            setu8(14);
            setu8(15);
        }
#undef setu8
        return p;
    }

    int fillArrayMaxMaskUint8SSE2(uint32_t * state, uint32_t * array,
                                  uint32_t max, uint32_t mask, int shift)
    {
        union w128xx_t {
            __m128i si128;
            uint8_t u8[16];
        } x;
        int p = 0;
        __m128i mask128 = _mm_set1_epi32(mask);
        __m128i shift128 = _mm_set1_epi32(shift);
        w128_t * state128 = reinterpret_cast<w128_t *>(state);
        //w128_t * array128 = reinterpret_cast<w128_t *>(array);
#define setu8(idx) array[p] = x.u8[idx]; p += (x.u8[idx] <= max) & 1
        for (int i = 0; i < size; i++) {
            x.si128 = _mm_srl_epi32(state128[i].si128, shift128);
            x.si128 = _mm_and_si128(x.si128, mask128);
//            array128[i].si128 = _mm_and_si128(x, mask128);
            setu8(0);
            setu8(1);
            setu8(2);
            setu8(3);
            setu8(4);
            setu8(5);
            setu8(6);
            setu8(7);
            setu8(8);
            setu8(9);
            setu8(10);
            setu8(11);
            setu8(12);
            setu8(13);
            setu8(14);
            setu8(15);
        }
#undef setu8
        return p;
    }
#endif
#else
    union w128_t {
        uint32_t u[4];
        uint64_t u64[2];
    };

    void fillState128(uint32_t *)
    {
        throw new std::logic_error("should not be called");
    }
    void fillArray128(uint32_t *, uint32_t *, int)
    {
        throw new std::logic_error("should not be called");
    }
#if 0
    void fillArrayMaxMaskSSE2(uint32_t *, uint32_t *, uint32_t, int)
    {
        throw new std::logic_error("should not be called");
    }
#endif
#endif // HAVE_SSE2

    inline static void lshift128(w128_t *out,  w128_t const *in, int shift);

    /**
     * This function simulates SIMD 128-bit right shift by the standard C.
     * The 128-bit integer given in in is shifted by (shift * 8) bits.
     * This function simulates the LITTLE ENDIAN SIMD.
     * @param out the output of this function
     * @param in the 128-bit data to be shifted
     * @param shift the shift value
     */
    inline static void rshift128(w128_t *out, w128_t const *in, int shift)
    {
        uint64_t th, tl, oh, ol;

        th = ((uint64_t)in->u[3] << 32) | ((uint64_t)in->u[2]);
        tl = ((uint64_t)in->u[1] << 32) | ((uint64_t)in->u[0]);

        oh = th >> (shift * 8);
        ol = tl >> (shift * 8);
        ol |= th << (64 - shift * 8);
        out->u[1] = (uint32_t)(ol >> 32);
        out->u[0] = (uint32_t)ol;
        out->u[3] = (uint32_t)(oh >> 32);
        out->u[2] = (uint32_t)oh;
    }

    /**
     * This function simulates SIMD 128-bit left shift by the standard C.
     * The 128-bit integer given in in is shifted by (shift * 8) bits.
     * This function simulates the LITTLE ENDIAN SIMD.
     * @param out the output of this function
     * @param in the 128-bit data to be shifted
     * @param shift the shift value
     */
    inline static void lshift128(w128_t *out, w128_t const *in, int shift)
    {
        uint64_t th, tl, oh, ol;

        th = ((uint64_t)in->u[3] << 32) | ((uint64_t)in->u[2]);
        tl = ((uint64_t)in->u[1] << 32) | ((uint64_t)in->u[0]);

        oh = th << (shift * 8);
        ol = tl << (shift * 8);
        oh |= tl >> (64 - shift * 8);
        out->u[1] = (uint32_t)(ol >> 32);
        out->u[0] = (uint32_t)ol;
        out->u[3] = (uint32_t)(oh >> 32);
        out->u[2] = (uint32_t)oh;
    }

    /**
     * This function represents the recursion formula.
     * @param r output
     * @param a a 128-bit part of the internal state array
     * @param b a 128-bit part of the internal state array
     * @param c a 128-bit part of the internal state array
     * @param d a 128-bit part of the internal state array
     */
    inline static void do_recursion(w128_t *r, w128_t *a, w128_t *b,
                                    w128_t *c, w128_t *d)
    {
        w128_t x;
        w128_t y;

        lshift128(&x, a, SFMT_SL2);
        rshift128(&y, c, SFMT_SR2);
        r->u[0] = a->u[0] ^ x.u[0] ^ ((b->u[0] >> SFMT_SR1) & SFMT_MSK1)
            ^ y.u[0] ^ (d->u[0] << SFMT_SL1);
        r->u[1] = a->u[1] ^ x.u[1] ^ ((b->u[1] >> SFMT_SR1) & SFMT_MSK2)
            ^ y.u[1] ^ (d->u[1] << SFMT_SL1);
        r->u[2] = a->u[2] ^ x.u[2] ^ ((b->u[2] >> SFMT_SR1) & SFMT_MSK3)
            ^ y.u[2] ^ (d->u[2] << SFMT_SL1);
        r->u[3] = a->u[3] ^ x.u[3] ^ ((b->u[3] >> SFMT_SR1) & SFMT_MSK4)
            ^ y.u[3] ^ (d->u[3] << SFMT_SL1);
    }

    /**
     * This function fills the internal state array with pseudorandom
     * integers.
     * @param sfmt SFMT internal state
     */
    void fillState32(uint32_t * state32)
    {
        int i;
        w128_t *r1, *r2;
        w128_t * state = reinterpret_cast<w128_t *>(state32);
        r1 = &state[SFMT_N - 2];
        r2 = &state[SFMT_N - 1];
        for (i = 0; i < SFMT_N - pos1; i++) {
            do_recursion(&state[i], &state[i],
                         &state[i + pos1], r1, r2);
            r1 = r2;
            r2 = &state[i];
        }
        for (; i < SFMT_N; i++) {
            do_recursion(&state[i], &state[i],
                         &state[i + pos1 - SFMT_N], r1, r2);
            r1 = r2;
            r2 = &state[i];
        }
    }

    void fillArray32(uint32_t * state32, uint32_t *array32, int length) {
        int i, j;
        w128_t *r1, *r2;
        DMSG("start fillArray32");

        w128_t * state = reinterpret_cast<w128_t *>(state32);
        w128_t * array = reinterpret_cast<w128_t *>(array32);
        r1 = &state[size - 2];
        r2 = &state[size - 1];
        for (i = 0; i < size - pos1; i++) {
            do_recursion(&array[i], &state[i], &state[i + pos1], r1, r2);
            r1 = r2;
            r2 = &array[i];
        }
        for (; i < size; i++) {
            do_recursion(&array[i], &state[i],
                         &array[i + pos1 - size], r1, r2);
            r1 = r2;
            r2 = &array[i];
        }
        for (; i < length; i++) {
            do_recursion(&array[i], &array[i - size],
                         &array[i + pos1 - size], r1, r2);
            r1 = r2;
            r2 = &array[i];
        }
        j = 0;
        for (i = length - size; i < length; i++) {
            state[j++] = array[i];
        }
    }

#if 0
    void fillArrayMaxMask(uint32_t * state,
                          uint32_t * array, uint32_t mask, int shift)
    {
        for (int i = 0; i < size32; i++) {
            uint32_t x = state[i] >> shift;
            array[i] = x & mask;
        }
    }
#endif
#if 0
    int fillArrayMaxInt8(uint8_t * state, uint8_t * array, uint8_t max)
    {
        int p = 0;
        uint8_t range = max + 1;
        uint8_t scale = UINT8_MAX / range;
        uint8_t over = scale * range;
        for (int i = 0; i < size32 * 4; i++) {
            array[p] = state[i] / scale;
            p += static_cast<int>(state[i] < over);
        }
        return p;
    }

    int fillArrayMaxInt16(uint16_t * state, uint16_t * array, uint16_t max)
    {
        int p = 0;
        uint16_t range = max + 1;
        uint16_t scale = UINT16_MAX / range;
        uint16_t over = scale * range;
        for (int i = 0; i < size32 * 2; i++) {
            array[p] = state[i] / scale;
            p += static_cast<int>(state[i] < over);
        }
        return p;
    }

    int fillArrayMaxInt32(uint32_t * state, uint32_t * array, uint32_t max)
    {
        int p = 0;
        uint32_t range = max + 1;
        uint32_t scale = UINT32_MAX / range;
        uint32_t over = scale * range;
        for (int i = 0; i < size32; i++) {
            array[p] = state[i] / scale;
            p += static_cast<int>(state[i] < over);
        }
        return p;
    }
#endif
#if 0
    int fillArrayMaxMin32(uint32_t * state, uint32_t * array,
                          uint32_t max, uint32_t min)
    {
        int p = 0;
        uint32_t range = static_cast<uint32_t>(max - min + 1);
        uint32_t scale = UINT32_MAX / range;
        uint32_t over = scale * range;
        for (int i = 0; i < size32; i++) {
            array[p] = state[i] / scale + min;
            p += static_cast<int>(state[i] < over);
        }
        return p;
    }
#endif
    /**
     * This function represents a function used in the initialization
     * by init_by_array
     * @param x 32-bit integer
     * @return 32-bit integer
     */
    uint32_t func1(uint32_t x)
    {
        return (x ^ (x >> 27)) * UINT32_C(1664525);
    }

    /**
     * This function represents a function used in the initialization
     * by init_by_array
     * @param x 32-bit integer
     * @return 32-bit integer
     */
    uint32_t func2(uint32_t x)
    {
        return (x ^ (x >> 27)) * UINT32_C(1566083941);
    }

    /**
     * This function certificate the period of 2^{MEXP}
     * @param sfmt SFMT internal state
     */
    void period_certification(uint32_t * psfmt32)
    {
        int inner = 0;
        int i, j;
        uint32_t work;
        const uint32_t parity[4] = {SFMT_PARITY1, SFMT_PARITY2,
                                    SFMT_PARITY3, SFMT_PARITY4};

        for (i = 0; i < 4; i++) {
            inner ^= psfmt32[idxof(i)] & parity[i];
        }
        for (i = 16; i > 0; i >>= 1) {
            inner ^= inner >> i;
        }
        inner &= 1;
        /* check OK */
        if (inner == 1) {
            return;
        }
        /* check NG, and modification */
        for (i = 0; i < 4; i++) {
            work = 1;
            for (j = 0; j < 32; j++) {
                if ((work & parity[i]) != 0) {
                    psfmt32[idxof(i)] ^= work;
                    return;
                }
                work = work << 1;
            }
        }
    }

}

namespace MersenneTwister {
    using namespace std;

    SFMT19937::SFMT19937(uint32_t seedValue)
    {
        cf = cpu_feature();
        state = alignedAlloc<uint32_t *>(stateSize32 * sizeof(uint32_t));
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue);
    }

    SFMT19937::SFMT19937(uint32_t seedValue[], int length)
    {
        cf = cpu_feature();
        state = alignedAlloc<uint32_t *>(stateSize32 * sizeof(uint32_t));
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue, length);
    }

    SFMT19937::~SFMT19937()
    {
        alignedFree(state);
    }

    const char * SFMT19937::getIDString()
    {
        return idString;
    }

    /**
     * This function initializes the internal state array with a 32-bit
     * integer seed.
     *
     * @param sfmt SFMT internal state
     * @param seed a 32-bit integer used as the seed.
     */
    void SFMT19937::seed(uint32_t seedValue)
    {
        int i;

        uint32_t *psfmt32 = state;

        psfmt32[idxof(0)] = seedValue;
        for (i = 1; i < stateSize32; i++) {
            psfmt32[idxof(i)] = 1812433253UL
                * (psfmt32[idxof(i - 1)]
                   ^ (psfmt32[idxof(i - 1)] >> 30))
                + i;
        }
        index = stateSize32;
        period_certification(state);
    }

    /**
     * This function initializes the internal state array,
     * with an array of 32-bit integers used as the seeds
     * @param sfmt SFMT internal state
     * @param init_key the array of 32-bit integers, used as a seed.
     * @param key_length the length of init_key.
     */
    void SFMT19937::seed(uint32_t *init_key, int key_length)
    {
        int i, j, count;
        uint32_t r;
        int lag;
        int mid;
        int size = stateSize32;
        uint32_t *psfmt32 = state;

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

        memset(state, 0x8b, sizeof(uint32_t) * stateSize32);
        if (key_length + 1 > stateSize32) {
            count = key_length + 1;
        } else {
            count = stateSize32;
        }
        r = func1(psfmt32[idxof(0)] ^ psfmt32[idxof(mid)]
                  ^ psfmt32[idxof(stateSize32 - 1)]);
        psfmt32[idxof(mid)] += r;
        r += key_length;
        psfmt32[idxof(mid + lag)] += r;
        psfmt32[idxof(0)] = r;

        count--;
        for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
            r = func1(psfmt32[idxof(i)]
                      ^ psfmt32[idxof((i + mid) % stateSize32)]
                      ^ psfmt32[idxof((i + stateSize32 - 1) % stateSize32)]);
            psfmt32[idxof((i + mid) % stateSize32)] += r;
            r += init_key[j] + i;
            psfmt32[idxof((i + mid + lag) % stateSize32)] += r;
            psfmt32[idxof(i)] = r;
            i = (i + 1) % stateSize32;
        }
        for (; j < count; j++) {
            r = func1(psfmt32[idxof(i)]
                      ^ psfmt32[idxof((i + mid) % stateSize32)]
                      ^ psfmt32[idxof((i + stateSize32 - 1) % stateSize32)]);
            psfmt32[idxof((i + mid) % stateSize32)] += r;
            r += i;
            psfmt32[idxof((i + mid + lag) % stateSize32)] += r;
            psfmt32[idxof(i)] = r;
            i = (i + 1) % stateSize32;
        }
        for (j = 0; j < stateSize32; j++) {
            r = func2(psfmt32[idxof(i)]
                      + psfmt32[idxof((i + mid) % stateSize32)]
                      + psfmt32[idxof((i + stateSize32 - 1) % stateSize32)]);
            psfmt32[idxof((i + mid) % stateSize32)] ^= r;
            r -= i;
            psfmt32[idxof((i + mid + lag) % stateSize32)] ^= r;
            psfmt32[idxof(i)] = r;
            i = (i + 1) % stateSize32;
        }

        index = stateSize32;
        period_certification(state);
    }

    void SFMT19937::fillState()
    {
        if (cf.avx2) {
            fillState256(state);
        } else if (cf.sse2) {
            fillState128(state);
        } else {
            fillState32(state);
        }
        index = 0;
    }

    void SFMT19937::fillArray(uint32_t * array, int length, bool skip)
    {
        DMSG("start fillArray");
        if (skip && index != stateSize32) {
            index = stateSize32;
        }
        if (length < stateSize32 || index != stateSize32 || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generate();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2 && (length % 8 == 0)) {
            DMSG("fillArray step 3");
            fillArray256(state, array, length / 4);
        } else if ((align % 16 == 0) && cf.sse2) {
            DMSG("fillArray step 3");
            fillArray128(state, array, length / 4);
        } else {
            DMSG("fillArray step 4");
            fillArray32(state, array, length / 4);
        }
        index = stateSize32;
    }

#if 0
//    int SFMT19937::fillArrayMaxInt(uint32_t * array,
//                                   uint32_t max, uint32_t mask, int shift)
    int SFMT19937::fillArrayMaxMin(uint32_t * array,
                                   uint32_t max, uint32_t min)
    {
#if defined(DEBUG)
        printf("max = %u\n", max);
        printf("min = %u\n", min);
#endif
        fillState();
        int r;
        r = fillArrayMaxMin32(state, array, max, min);
#if defined(DEBUG)
        printf("r = %d\n", r);
#endif
        return r;
    }
#endif
}
