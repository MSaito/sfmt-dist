#pragma once
#ifndef MT_DSFMTAVX_HPP
#define MT_DSFMTAVX_HPP
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <inttypes.h>
#include <sfmt-dist/aligned_alloc.h>
#include <sfmt-dist/cpu_feature.h>
#include "sfmt_common.h"
#include "dsfmt_avx.h"
#include "debug.h"

namespace MersenneTwister {
    template<int mersenneExponent>
    class DSFMTAVX {
    private:
        double * state;
        dSFMT_AVX::params *params;
        int index;
        cpu_feature_t cf;
        int stateSize;
        int stateSize64;
        void fillState();
        void init() {
            cf = cpu_feature();
            params = dSFMT_AVX::search_params(mersenneExponent);
            if (params == NULL) {
                throw new std::runtime_error("wrong Mersenne Exponent");
            }
            stateSize = params->array_size;
            stateSize64 = stateSize * 4;
            size_t alloc_size = (stateSize + 1) * 4 * sizeof(uint64_t);
            state = alignedAlloc<double *>(alloc_size);
            if (state == NULL) {
                throw new std::runtime_error("can't get aligned memory");
            }
        }
    public:
#if __cplusplus >= 201103L
        using result_type = double;
        using seed_type = uint32_t;
#else
        typedef double result_type;
        typedef uint32_t seed_type;
#endif
        DSFMTAVX(uint32_t seedValue = 1234) {
            init();
            seed(seedValue);
        }
        DSFMTAVX(uint32_t seedValue[], int length) {
            init();
            seed(seedValue, length);
        }
        ~DSFMTAVX() {
            alignedFree(state);
        }
        void seed(uint32_t seedValue) {
            sfmt_seed(reinterpret_cast<w256_t *>(state),
                      stateSize + 1, seedValue);
            w256_t * pstate = reinterpret_cast<w256_t *>(state);
            sfmt_period_certification(&pstate[stateSize], &params->pcv);
            dSFMT_AVX::initial_mask(pstate, stateSize);
            index = stateSize64;
        }
        void seed(uint32_t seedValue[], int length) {
            sfmt_seed(reinterpret_cast<w256_t *>(state),
                      stateSize + 1, seedValue, length);
            w256_t * pstate = reinterpret_cast<w256_t *>(state);
            sfmt_period_certification(&pstate[stateSize], &params->pcv);
            dSFMT_AVX::initial_mask(pstate, stateSize);
            index = stateSize64;
        }
        int getMersenneExponent() {
            return mersenneExponent;
        }
        int blockSize() {
            return stateSize64;
        }
        double generate() {
            return generateClose1Open2() - 1.0;
        }
        double operator()() {
            return generate();
        }
        double generateClose1Open2() {
            if (index >= stateSize64) {
                fillState();
                index = 0;
            }
            return state[index++];
        }
        double generateClose0Open1() {
            return generateClose1Open2() - 1.0;
        }
        double generateOpen0Close1() {
            return 2.0 - generateClose1Open2();
        }
        double generateOpen0Open1() {
            union {
                double d;
                uint64_t u;
            } x;
            x.d = generateClose1Open2();
            x.u |= 1;
            return x.d - 1.0;
        }
        const std::string getIDString() {
            return dSFMT_AVX::get_id_string(*params);
        }
        void fillArray(double * array, int length);
        void fillArrayClose0Open1(double * array, int length) {
            fillArray(array, length);
        }
        void fillArrayClose1Open2(double * array, int length);
        void fillArrayOpen0Close1(double * array, int length);
        void fillArrayOpen0Open1(double * array, int length);
        void fillArrayMaxInt(int32_t * array, uint64_t rmax, int32_t min);
        int fillArrayNormalDist(double * array, double mu, double sigma);
        bool selfTest();
#if defined(DEBUG)
        void d_p() {
            cout << "debug_print" << endl;
            cout << "array_size = " << dec << params->array_size << endl;
            cout << "mexp = " << dec << params->mexp << endl;
            cout << "pos1 = " << dec << params->pos1 << endl;
            cout << "mask[0] = " << hex << params->mask.u64[0] << endl;
            cout << "mask[1] = " << hex << params->mask.u64[1] << endl;
            cout << "mask[2] = " << hex << params->mask.u64[2] << endl;
            cout << "mask[3] = " << hex << params->mask.u64[3] << endl;
            w256_t * state256 = reinterpret_cast<w256_t *>(state);
            for (int i = 0; i < stateSize + 1; i++) {
                for (int j = 0; j < 4; j++) {
                    cout << hex << setfill('0') << setw(16)
                         << state256[i].u64[j]
                         << " ";
                }
                cout << endl;
            }
        }
#endif
    };

    template<int mersenneExponent>
    inline void DSFMTAVX<mersenneExponent>::fillState()
    {
        dSFMT_AVX::fill_state(cf, *params, state);
        index = 0;
    }

    template<int mersenneExponent>
    inline void
    DSFMTAVX<mersenneExponent>::fillArray(double * array,
                                          int length)
    {
        if (!HAVE_AVX2
            || length < stateSize64
            || index != stateSize64
            || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generate();
            }
            return;
        }
        dSFMT_AVX::fill_array(cf, *params, state, array, length / 4);
    }

    template<int mersenneExponent>
    inline void
    DSFMTAVX<mersenneExponent>::fillArrayClose1Open2(double * array,
                                                     int length)
    {
        if (!HAVE_AVX2
            || length < stateSize64
            || index != stateSize64
            || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateClose1Open2();
            }
            return;
        }
        dSFMT_AVX::fill_arrayc1o2(cf, *params, state, array, length / 4);
    }

    template<int mersenneExponent>
    inline void
    DSFMTAVX<mersenneExponent>::fillArrayOpen0Close1(double * array,
                                                     int length)
    {
        if (!HAVE_AVX2
            || length < stateSize64
            || index != stateSize64
            || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateOpen0Close1();
            }
            return;
        }
        dSFMT_AVX::fill_arrayo0c1(cf, *params, state, array, length / 4);
    }

    template<int mersenneExponent>
    inline void
    DSFMTAVX<mersenneExponent>::fillArrayOpen0Open1(double * array,
                                                    int length)
    {
        if (!HAVE_AVX2
            || length < stateSize64
            || index != stateSize64
            || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateOpen0Open1();
            }
            return;
        }
        dSFMT_AVX::fill_arrayo0o1(cf, *params, state, array, length / 4);
    }

    template<int mersenneExponent>
    inline void DSFMTAVX<mersenneExponent>::fillArrayMaxInt(int32_t * array,
                                                            uint64_t rmax,
                                                            int32_t min)
    {
        dSFMT_AVX::fillarray_maxint(cf, *params, state, array, rmax, min);
    }

    template<int mersenneExponent>
    inline int DSFMTAVX<mersenneExponent>::fillArrayNormalDist(double * array,
                                                                double mu,
                                                                double sigma)
    {
        return dSFMT_AVX::fillarray_normaldist(cf, *params,
                                               state, array, mu, sigma);
    }

    template<int mersenneExponent>
    inline bool DSFMTAVX<mersenneExponent>::selfTest()
    {
        return dSFMT_AVX::self_test(cf, *params, state);
    }

#define MT_DSFMTAVX_FILLSTATE(avx_mexp)                         \
    template<>                                                  \
    inline void                                                 \
    DSFMTAVX<avx_mexp>::fillState()                             \
    {                                                           \
        if (HAVE_AVX2 && cf.avx2) {                             \
            dSFMT_AVX::fill_state256_ ## avx_mexp (state);      \
        } else {                                                \
            dSFMT_AVX::fill_state64_ ## avx_mexp (state);       \
        }                                                       \
        index = 0;                                              \
    }

    MT_DSFMTAVX_FILLSTATE(607)
    MT_DSFMTAVX_FILLSTATE(1279)
    MT_DSFMTAVX_FILLSTATE(2281)
    MT_DSFMTAVX_FILLSTATE(4253)
    MT_DSFMTAVX_FILLSTATE(19937)
#undef MT_DSFMTAVX_FILLSTATE

#define MT_DSFMTAVX_FILLARRAY(fn0, fn1, fn2, avx_mexp)          \
    template<>                                                  \
    inline void                                                 \
    DSFMTAVX<avx_mexp>::fn0 (double * array, int length)        \
    {                                                           \
        DMSG("start fillArray");                                \
        if (!HAVE_AVX2                                          \
            || length < stateSize64                             \
            || index != stateSize64                             \
            || (length % 4 != 0)) {                             \
            for (int i = 0; i < length; i++) {                  \
                array[i] = fn1 ();                              \
            }                                                   \
            return;                                             \
        }                                                       \
        int align = reinterpret_cast<uintptr_t>(array) % 32;    \
        if (HAVE_AVX2 && cf.avx2 && (align == 0)) {             \
            dSFMT_AVX::                                         \
                fill_array256_ ## fn2 ## _ ## avx_mexp          \
                (state, array, length / 4);                     \
        } else {                                                \
            dSFMT_AVX::                                         \
                fill_array64_ ## fn2 ## _ ## avx_mexp           \
                (state, array, length / 4);                     \
        }                                                       \
    }

    MT_DSFMTAVX_FILLARRAY(fillArray, generate, c0o1, 607)
    MT_DSFMTAVX_FILLARRAY(fillArray, generate, c0o1, 1279)
    MT_DSFMTAVX_FILLARRAY(fillArray, generate, c0o1, 2281)
    MT_DSFMTAVX_FILLARRAY(fillArray, generate, c0o1, 4253)
    MT_DSFMTAVX_FILLARRAY(fillArray, generate, c0o1, 19937)
    MT_DSFMTAVX_FILLARRAY(fillArrayClose1Open2, generateClose1Open2,
                           c1o2, 607)
    MT_DSFMTAVX_FILLARRAY(fillArrayClose1Open2, generateClose1Open2,
                           c1o2, 1279)
    MT_DSFMTAVX_FILLARRAY(fillArrayClose1Open2, generateClose1Open2,
                           c1o2, 2281)
    MT_DSFMTAVX_FILLARRAY(fillArrayClose1Open2, generateClose1Open2,
                           c1o2, 4253)
    MT_DSFMTAVX_FILLARRAY(fillArrayClose1Open2, generateClose1Open2,
                           c1o2, 19937)
    MT_DSFMTAVX_FILLARRAY(fillArrayOpen0Close1, generateOpen0Close1,
                           o0c1, 607)
    MT_DSFMTAVX_FILLARRAY(fillArrayOpen0Close1, generateOpen0Close1,
                           o0c1, 1279)
    MT_DSFMTAVX_FILLARRAY(fillArrayOpen0Close1, generateOpen0Close1,
                           o0c1, 2281)
    MT_DSFMTAVX_FILLARRAY(fillArrayOpen0Close1, generateOpen0Close1,
                           o0c1, 4253)
    MT_DSFMTAVX_FILLARRAY(fillArrayOpen0Close1, generateOpen0Close1,
                           o0c1, 19937)
    MT_DSFMTAVX_FILLARRAY(fillArrayOpen0Open1, generateOpen0Open1,
                           o0o1, 607)
    MT_DSFMTAVX_FILLARRAY(fillArrayOpen0Open1, generateOpen0Open1,
                           o0o1, 1279)
    MT_DSFMTAVX_FILLARRAY(fillArrayOpen0Open1, generateOpen0Open1,
                           o0o1, 2281)
    MT_DSFMTAVX_FILLARRAY(fillArrayOpen0Open1, generateOpen0Open1,
                           o0o1, 4253)
    MT_DSFMTAVX_FILLARRAY(fillArrayOpen0Open1, generateOpen0Open1,
                           o0o1, 19937)
#undef MT_DSFMTAVX_FILLARRAY

#define MT_DSFMTAVX_FILLARRAYMAXINT(avx_mexp)                   \
    template<>                                                  \
    inline void                                                 \
    DSFMTAVX<avx_mexp>::fillArrayMaxInt(int32_t *array,         \
                                        uint64_t rmax,          \
                                        int32_t min)            \
    {                                                           \
        if (cf.avx2) {                                          \
            dSFMT_AVX::fillarray256_maxint_ ## avx_mexp (state, \
                                                      array,    \
                                                      rmax,     \
                                                      min);     \
        } else {                                                \
            dSFMT_AVX::fillarray64_maxint_ ## avx_mexp (state,  \
                                                      array,    \
                                                      rmax,     \
                                                      min);     \
        }                                                       \
    }
    MT_DSFMTAVX_FILLARRAYMAXINT(19937)
#undef MT_DSFMTAVX_FILLARRAYMAXINT

#define MT_DSFMTAVX_FILLARRAYNORMALDIST(avx_mexp)               \
    template<>                                                  \
    inline int                                                 \
    DSFMTAVX<avx_mexp>::fillArrayNormalDist(double *array,      \
                                            double mu,          \
                                            double sigma)       \
    {                                                           \
        if (cf.avx2) {                                          \
            return dSFMT_AVX::fillarray256_boxmuller_ ## avx_mexp \
                (state,    \
                                                      array,    \
                                                      mu,     \
                                                      sigma);     \
        } else {                                                \
            return dSFMT_AVX::fillarray64_boxmuller_ ## avx_mexp \
               (state,    \
                                                      array,    \
                                                      mu,     \
                                                      sigma);     \
        }                                                       \
    }
    MT_DSFMTAVX_FILLARRAYNORMALDIST(19937)
#undef MT_DSFMTAVX_FILLARRAYNORMALDIST

#define MT_DSFMTAVX_SELFTEST(avx_mexp)                          \
    template<>                                                  \
    inline bool                                                 \
    DSFMTAVX<avx_mexp>::selfTest()                              \
    {                                                           \
        return dSFMT_AVX::self_test_ ## avx_mexp (cf, *params,  \
                                                  state);       \
    }
    MT_DSFMTAVX_SELFTEST(19937)
#undef MT_DSFMTAVX_SELFTEST
}
#endif // MT_DSFMTAVX_H
