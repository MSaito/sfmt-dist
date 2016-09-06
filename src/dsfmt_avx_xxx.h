#include "config.h"
#include <stdexcept>
#include "dsfmt_avx.h"
#include <sfmt-dist/cpu_feature.h>
#include "w256.h"
#include "debug.h"

#if HAVE_ZMMINTRIN_H
#include <zmmintrin.h>
#endif

#if HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif

namespace {
    using namespace MersenneTwister;
    using namespace std;

#define DSFMTAVX_SL1 19
#define DSFMTAVX_SR 12

#if DSFMT_MEXP == 1279
#define DSFMTAVX_SIZE 5
#define DSFMTAVX_POS1 3
    const w256_t mask1 = {{UINT64_C(0x000ff9ff37fefbbf),
                           UINT64_C(0x0007bebd2cebad7c),
                           UINT64_C(0x000bfffbcfaddfce),
                           UINT64_C(0x0005affefd978fff)}};
    const w256_t fix1 = {{UINT64_C(0xfbc5b6c65831cf64),
                          UINT64_C(0x328ec06650089455),
                          UINT64_C(0xa6ea0cee0217b775),
                          UINT64_C(0x119ade137f700f73)}};
    const w256_t pcv1 = {{UINT64_C(0x0000000000000001),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0xb6c4200000000000)}};
#endif // 1279
#if DSFMT_MEXP == 2281
#define DSFMTAVX_SIZE 10
#define DSFMTAVX_POS1 5
    const w256_t mask1 = {{UINT64_C(0x00053fdfdbdecff4),
                           UINT64_C(0x000fd5edfbdd7fb7),
                           UINT64_C(0x000ffcbfb797f6f5),
                           UINT64_C(0x000f69effd89efac)}};
    const w256_t fix1 = {{UINT64_C(0x7aee281d3142542a),
                          UINT64_C(0x176a4fb0881dafba),
                          UINT64_C(0x13e626916da50640),
                          UINT64_C(0x65b20b60bd5c0674)}};
    const w256_t pcv1 = {{UINT64_C(0x0000000000000001),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0xdbf3c390a41d2200)}};
#endif // 2281
#if DSFMT_MEXP == 4253
#define DSFMTAVX_SIZE 20
#define DSFMTAVX_POS1 9
    const w256_t mask1 = {{UINT64_C(0x000a9feff9ebff9f),
                           UINT64_C(0x000efeffbbf3777f),
                           UINT64_C(0x0001bfffdbfdf7bf),
                           UINT64_C(0x000c7e3fd779bded)}};
    const w256_t fix1 = {{UINT64_C(0xa0cfe3b5ab1db41d),
                          UINT64_C(0xb34f9a51d50878c4),
                          UINT64_C(0xabebbce7898197fe),
                          UINT64_C(0x508ddb4ba74644bf)}};
    const w256_t pcv1 = {{UINT64_C(0x0000000000000001),
                          UINT64_C(0x2ada8a6cc0000000),
                          UINT64_C(0xf566075647a7b032),
                          UINT64_C(0x4a8f059463e81559)}};
#endif // 4253
#if DSFMT_MEXP == 19937
#define DSFMTAVX_SIZE 95
#define DSFMTAVX_POS1 47
    const w256_t mask1 = {{UINT64_C(0x000f7eefaefbd7e9),
                           UINT64_C(0x000cd7fe2ffcfcc3),
                           UINT64_C(0x000ff2fdf7fab37f),
                           UINT64_C(0x000cffffd6adff3c)}};
    const w256_t fix1 = {{UINT64_C(0x42e415394eb76145),
                          UINT64_C(0xb1a401c199de27fd),
                          UINT64_C(0x379b5a55f2680707),
                          UINT64_C(0x1788815e1a4a4cdd)}};
    const w256_t pcv1 = {{UINT64_C(0x0000000001000001),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0x7c32000000000000),
                          UINT64_C(0x4f259872bc33ab2d)}};
#endif // 19937

#include "dsfmt_avx_common.h"

#define CON(a, b) CON2(a, b)
#define CON2(a, b) a ## b

#if HAVE_AVX2 && HAVE_IMMINTRIN_H
    const w256x32_t perm = {{7, 0, 1, 2, 3, 4, 5, 6}};
    const w256_t one = {{1, 1, 1, 1}};
    const w256xd_t m_one = {{-1.0, -1.0, -1.0, -1.0}};
    const w256xd_t two = {{2.0, 2.0, 2.0, 2.0}};
#endif
}

namespace MersenneTwister {
    namespace dSFMT_AVX {
#if HAVE_AVX2 && HAVE_IMMINTRIN_H
        /**
         * This function fills the internal state array with pseudorandom
         * integers.
         * @param p parameter
         * @param state SFMT internal state
         */
        void CON(fillState256_, DSFMT_MEXP)(double * state) {
            w256_t * pstate = reinterpret_cast<w256_t *>(state);
            __m256i lung = pstate[DSFMTAVX_SIZE].si256;
            __m256i mask = mask1.si256;
            int i = 0;
            for (;i < DSFMTAVX_SIZE - DSFMT_POS1; i++) {
                pstate[i].si256 = recursion256(mask,
                                               pstate[i].si256,
                                               pstate[i + DSFMT_POS1],
                                               &lung);
            }
            for (;i < DSFMTAVX_SIZE; i++) {
                pstate[i].si256
                    = recursion256(mask,
                                   pstate[i].si256,
                                   pstate[i + DSFMTAVX_POS1 - DSFMTAVX_SIZE],
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
        void CON(fillArray256_c1o2_, DSFMT_MEXP) (double * state,
                                                  double * array64, int length)
        {
            w256_t * pstate = reinterpret_cast<w256_t *>(state);
            w256_t * array = reinterpret_cast<w256_t *>(array64);
            const __m256i mask = mask1.si256;
            __m256i lung = pstate[2].si256;
            __m256i a = pstate[0].si256;
            __m256i b = pstate[1].si256;
            int i = 0;
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
                array[i].si256 = recursion256(mask,
                                              pstate[i].256,
                                              pstate[i + DSFMTAVX_POS1].256,
                                              &lung);
            }
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
                array[i].si256
                    = recursion256(mask,
                                   pstate[i].256,
                                   array[i + DSFMTAVX_POS1-DSFMTAVX_SIZE].256,
                                   &lung);
            }
            for (; i < length; i++) {
                array[i].si256
                    = recursion256(mask,
                                   array[i - DSFMTAVX_SIZE].256,
                                   array[i + DSFMTAVX_POS1-DSFMTAVX_SIZE].256,
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

#define FILLARRAY256(fn1, fn2)                                          \
        void fn1 (double * state, double * array64, int length)         \
        {                                                               \
            w256_t * pstate = reinterpret_cast<w256_t *>(state);        \
            w256_t * array = reinterpret_cast<w256_t *>(array64);       \
            const __m256i mask = mask1.si256;                           \
            __m256i lung = pstate[2].si256;                             \
            __m256i a = pstate[0].si256;                                \
            __m256i b = pstate[1].si256;                                \
            int i = 0;                                                  \
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {            \
                array[i].si256 = recursion256(mask,                     \
                                              pstate[i].256,            \
                                              pstate[i + DSFMTAVX_POS1].256, \
                                              &lung);                   \
            }                                                           \
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {            \
                array[i].si256                                          \
                    = recursion256(mask,                                \
                                   pstate[i].256,                       \
                                   array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].256, \
                                   &lung);                              \
            }                                                           \
            for (; i < length; i++) {                                   \
                array[i].si256                                          \
                    = recursion256(mask,                                \
                                   array[i - DSFMTAVX_SIZE].256,       \
                                   array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].256, \
                                   &lung);                              \
                fn2 (&array[i - DSFMTAVX_SIZE]);                        \
            }                                                           \
            int j = 0;                                                  \
            for (int i = length - DSFMTAVX_SIZE; i < length ;i++) {     \
                pstate[j++].si256 = array[i].si256;                     \
                fn2 (&array[i]);                                        \
            }                                                           \
            pstate[DSFMTAVX_SIZE].si256 = lung;                         \
            _mm256_zeroall();                                           \
        }

        FILLARRAY256(CON(fillArray256_o0c1_, DSFMT_MEXP), convert256_c0o1)
        FILLARRAY256(CON(fillArray256_o0c1_, DSFMT_MEXP), convert256_o0c1)
        FILLARRAY256(CON(fillArray256_o0c1_, DSFMT_MEXP), convert256_o0o1)

#else // don't HAVE_AVX
        void CON(fillState256_, DSFMT_MEXP) (double *)
        {
            throw new std::logic_error("should not be called");
        }

        void CON(fillArray256_c1o2_, DSFMT_MEXP) (double *, double *, int)
        {
            throw new std::logic_error("should not be called");
        }
        void CON(fillArray256_c0o1_, DSFMT_MEXP) (double *, double *, int)
        {
            throw new std::logic_error("should not be called");
        }
        void CON(fillArray256_o0c1_, DSFMT_MEXP) (double *, double *, int)
        {
            throw new std::logic_error("should not be called");
        }
        void CON(fillArray256_o0o1_, DSFMT_MEXP) (double *, double *, int)
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
        void CON(fillState64_, DSFMT_MEXP) (double * state64)
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

        void CON(fillArray64_c1o2_, DSFMT_MEXP) (double * state64,
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

#define FILLARRAY64(fn1, fn2)                                           \
        void fn1 (double * state64, double * array64, int length)       \
        {                                                               \
            w256_t * state = reinterpret_cast<w256_t *>(state64);       \
            w256_t * array = reinterpret_cast<w256_t *>(array64);       \
            w256_t *lung = &state[2];                                   \
            int i = 0;                                                  \
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {            \
                recursion64(&array[i], &state[i],                       \
                            &state[i + DSFMTAVX_POS1], lung);           \
            }                                                           \
            for (; i < DSFMTAVX_SIZE; i++) {                            \
                recursion64(&array[i], &state[i],                       \
                            &array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE], lung); \
            }                                                           \
            for (; i < length; i++) {                                   \
                recursion64(&array[i], &array[i - DSFMTAVX_SIZE],       \
                            &array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE], lung); \
                fn2 (&array[i - DSFMTAVX_SIZE]);                        \
            }                                                           \
            int j = 0;                                                  \
            for (i = length - DSFMTAVX_SIZE; i < length; i++) {         \
                state[j++] = array[i];                                  \
                fn2 (&array[i]);                                        \
            }                                                           \
            state[DSFMTAVX_SIZE] = *lung;                               \
        }

        FILLARRAY64(CON(fillArray64_c0o1_, DSFMT_MEXP), convert64_c0o1)
        FILLARRAY64(CON(fillArray64_o0c1_, DSFMT_MEXP), convert64_o0c1)
        FILLARRAY64(CON(fillArray64_o0o1_, DSFMT_MEXP), convert64_o0o1)
    }
}


#if 0
namespace MersenneTwister {
    using namespace std;

#define CON(a, b) CON2(a, b)
#define CON2(a, b) a ## b

    void CON(dsfmtavx2_fill_state, DSFMT_MEXP)(const cpu_feature_t& cf,
                                               double * state)
    {
        if (HAVE_AVX2 && cf.avx2) {
            fillState256(state);
        } else {
            fillState64(state);
        }
    }

    void CON(dsfmtavx2_fill_array, DSFMT_MEXP)(const cpu_feature_t& cf,
                                               double * state,
                                               double * array,
                                               int length)
    {
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if (HAVE_AVX2 && cf.avx2 && (align == 0)) {
            fillArray256_c0o1(state, array, length / 4);
        } else {
            fillArray64_c0o1(state, array, length / 4);
        }
    }

    void CON(dsfmtavx2_fill_arrayc1o2, DSFMT_MEXP)(const cpu_feature_t& cf,
                                                   double * state,
                                                   double * array,
                                                   int length)
    {
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if (HAVE_AVX2 && cf.avx2 && (align == 0)) {
            fillArray256_c1o2(state, array, length / 4);
        } else {
            fillArray64_c1o2(state, array, length / 4);
        }
    }

    void CON(dsfmtavx2_fill_arrayo0c1, DSFMT_MEXP)(const cpu_feature_t& cf,
                                                   double * state,
                                                   double * array,
                                                   int length)
    {
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if (HAVE_AVX2 && cf.avx2 && (align == 0)) {
            fillArray256_o0c1(state, array, length / 4);
        } else {
            fillArray64_o0c1(state, array, length / 4);
        }
    }

    void CON(dsfmtavx2_fill_arrayo0o1, DSFMT_MEXP)(const cpu_feature_t& cf,
                                                   double * state,
                                                   double * array,
                                                   int length)
    {
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if (HAVE_AVX2 && cf.avx2 && (align == 0)) {
            fillArray256_o0o1(state, array, length / 4);
        } else {
            fillArray64_o0o1(state, array, length / 4);
        }
    }
}
#endif
